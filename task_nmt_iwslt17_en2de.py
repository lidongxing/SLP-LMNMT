#! -*- coding: utf-8 -*-
# bert做NMT任务，采用UNILM方案
#参考bert4keras：https://github.com/bojone/bert4keras
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
#预处理multi-language BERT
config_path = '/models/multi_cased_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/models/multi_cased_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/models/multi_cased_L-12_H-768_A-12/vocab.txt'



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
train_data = load_data('data/ende/corpus_deen.tsv')
valid_data = load_data('data/ende/dev2010_deen.tsv')
#test_data = load_data('data/ende/test2010.tsv')

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
                content, title, maxlen=maxlen
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
            model.save_weights('./train_models/best_model_deen.weights')

#train
if __name__ == '__main__':

     evaluator = Evaluator()
     train_generator = data_generator(train_data, batch_size)
     # d = train_generator.forfit()
     # print(d.__next__())
     # print(d.__next__())

     model.fit(
         train_generator.forfit(),
         steps_per_epoch=len(train_generator),
         epochs=epochs,
         callbacks=[evaluator]
     )
#
else:
    model.load_weights('./train_models/best_model_deen.weights')

# '''''''''''''''''''''test'''''''''''''''''
model.load_weights('./train_models/best_model_deen.weights')
dess = []
for (de,en) in valid_data:
    des = autotitle.generate(en,topk=4)
    dess.append(des)

import codecs
def write_trans_result(filename):
    with codecs.open(filename,'w') as f:
        f.write('\n'.join(dess))
write_trans_result('datasets/iwslt2017/ende/dev2010.de.trans')


#---------------------get source file-------------
import codecs
def get_src(filename1,filename2):
#     """加载数据
#     单条格式：(标题, 正文)
#     """
     D = []
     with open(filename1) as f:
         for l in f:
             title, content = l.strip().split('\t')
             D.append(title)
     with open(filename2,'w') as f1:
         f1.write('\n'.join(D))

get_src('datasets/iwslt2017/ende/dev2010_deen.tsv','datasets/iwslt2017/ende/dev2010.src.en')

