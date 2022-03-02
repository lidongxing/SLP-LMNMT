#! -*- coding: utf-8 -*-
# bert做Seq2Seq任务，采用UNILM方案
# 介绍链接：https://kexue.fm/archives/6933
# 数据集：https://github.com/CLUEbenchmark/CLGE 中的CSL数据集
# 补充了评测指标bleu、rouge-1、rouge-2、rouge-l
from __future__ import print_function
import os
os.environ['TF_KERAS']= '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
from tqdm import tqdm
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, open
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from keras.models import Model
from rouge import Rouge  # pip install rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import tensorflow as tf


# 基本参数
maxlen = 200
batch_size = 32
epochs = 8

# bert配置
config_path = '/data/lidongxing/bert4keras-master/models/multi_cased_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/data/lidongxing/bert4keras-master/models/multi_cased_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/data/lidongxing/bert4keras-master/models/multi_cased_L-12_H-768_A-12/vocab.txt'



def load_data(filename):
    """加载数据
    单条格式：(标题, 正文)
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            title, content = l.strip().split('\t')
            D.append((title, content))
    return D


# 加载数据集
train_data = load_data('datasets/iwslt2017/ende/corpus_deen.tsv')
valid_data = load_data('datasets/iwslt2017/ende/dev2010_deen.tsv')
#test_data = load_data('datasets/iwslt2016/test_lowcased.tsv')

# 加载并精简词表，建立分词器
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
# token_dict = load_vocab(dict_path)
tokenizer = Tokenizer(token_dict, do_lower_case=True)

class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        for is_end, (title, content) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(
                title, content, maxlen=maxlen
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield ([batch_token_ids, batch_segment_ids],None )
                batch_token_ids, batch_segment_ids = [], []


class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分
    """
    def compute_loss(self, inputs, mask=None):
        y_true, y_mask, y_pred = inputs
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = y_mask[:, 1:]  # segment_ids，刚好指示了要预测的部分
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss

# strategy = tf.distribute.MirroredStrategy()  # 建立单机多卡策略
#
# with strategy.scope():  # 调用该策略
model = build_transformer_model(
    config_path,
    checkpoint_path,
    application='unilm',
    # keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
)

output = CrossEntropy(2)(model.inputs + model.outputs)
model = Model(model.inputs, output)
model.compile(optimizer=Adam(1e-5))
model.summary()

class AutoTitle(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        return self.last_token(model).predict([token_ids, segment_ids])

    def generate(self, text, topk=1):
        max_c_len = maxlen - self.maxlen
        token_ids, segment_ids = tokenizer.encode(text, maxlen=max_c_len)
        output_ids = self.beam_search([token_ids, segment_ids],
                                      topk=topk)  # 基于beam search
        return tokenizer.decode(output_ids)


autotitle = AutoTitle(start_id=None, end_id=tokenizer._token_end_id, maxlen=64) #32,64,80   64 best


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.lowest = 1e10

    def on_epoch_end(self, epoch, logs=None):
        # 保存最优
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            model.save_weights('./iwslt2017_de2en_model/best_model_deen.weights')


#if __name__ == '__main__':

#     evaluator = Evaluator()
#     train_generator = data_generator(train_data, batch_size)
     # d = train_generator.forfit()
     # print(d.__next__())
     # print(d.__next__())

#     model.fit(
#         train_generator.forfit(),
#         steps_per_epoch=len(train_generator),
#         epochs=epochs,
#         callbacks=[evaluator]
#     )
#
#else:

#     model.load_weights('./iwslt2017_de2en_model/best_model_deen.weights')

# '''''''''''''''''''''test'''''''''''''''''
model.load_weights('./iwslt2017_de2en_model/best_model_deen.weights')
#enss = []
#for (de,en) in valid_data:
#    ens = autotitle.generate(de,topk=4)
#    enss.append(ens)

ens = autotitle.generate("Wir müssen einen weiblichen Diskurs entwickeln, der die folgenden Werte nicht nur würdigt, sondern auch umsetzt: Gnade anstatt Rache, Zusammenarbeit anstatt Konkurrenz, Einschluss anstatt Ausschluss.",topk=4)
print(ens)

#import codecs
#def write_trans_result(filename):
#    with codecs.open(filename,'w') as f:
#        f.write('\n'.join(enss))
#write_trans_result('datasets/iwslt2017/ende/dev2010.en.trans')


#---------------------get source file-------------
#import codecs
#def get_src(filename1,filename2):
#     """加载数据
#     单条格式：(标题, 正文)
#     """
#     D = []
#     with open(filename1) as f:
#         for l in f:
#             title, content = l.strip().split('\t')
#             D.append(content)
#     with open(filename2,'w') as f1:
#         f1.write('\n'.join(D))

#get_src('datasets/iwslt2017/ende/dev2010_deen.tsv','datasets/iwslt2017/ende/dev2010.src.en')
#en2de BLEU:5.09
#de2en BLEU:14.18
#德语、荷兰-日耳曼语系
#法语、西班牙、葡萄牙、罗马尼亚-拉丁语系
