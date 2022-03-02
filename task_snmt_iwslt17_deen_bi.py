#! -*- coding: utf-8 -*-
# bert做Seq2Seq任务，采用UNILM方案
# 介绍链接：https://kexue.fm/archives/6933
# 数据集：https://github.com/CLUEbenchmark/CLGE 中的CSL数据集
# 补充了评测指标bleu、rouge-1、rouge-2、rouge-l
from __future__ import print_function
import os
os.environ['TF_KERAS']= '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import numpy as np
from tqdm import tqdm
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam,is_tf_keras
from bert4keras.snippets import sequence_padding, open
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from bert4keras.optimizers import extend_with_weight_decay
from rouge import Rouge  # pip install rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import random as rand
import tensorflow as tf

# 基本参数
maxlen = 128
batch_size = 32
epochs = 8

# bert配置
config_path = '/data/lidongxing/bert4keras-master3/models/multi_cased_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/data/lidongxing/bert4keras-master3/models/multi_cased_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/data/lidongxing/bert4keras-master3/models/multi_cased_L-12_H-768_A-12/vocab.txt'



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
valid_data = load_data('datasets/iwslt2017/ende/test2010_deen.tsv')
#test_data = load_data('/data/lidongxing/bert4keras-master/examples/datasets/iwslt2016/test_lowcased.tsv')

# 加载并精简词表，建立分词器
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
# tokenizer = Tokenizer(token_dict, do_lower_case=True)
# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_input_ids, batch_segment_ids,batch_labels,batch_token_output_ids = [], [],[],[]
        for is_end, (title, content) in self.sample(random):
            if rand.randint(1,10)>5:
                token_ids, segment_ids = tokenizer.encode(
                    title, content, maxlen=maxlen
                )
                label = 0 #englist
            else:
                token_ids, segment_ids = tokenizer.encode(
                    content, title, maxlen=maxlen
                )
                label = 1  # cs
            batch_token_input_ids.append(token_ids[:-1])
            batch_token_output_ids.append(token_ids[1:])
            batch_segment_ids.append(segment_ids[:-1])
            batch_labels.append([label])
            if len(batch_token_input_ids) == self.batch_size or is_end:
                batch_token_input_ids = sequence_padding(batch_token_input_ids)
                batch_token_output_ids = sequence_padding(batch_token_output_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = np.array(batch_labels)
                yield ([batch_token_input_ids, batch_segment_ids], [batch_labels,batch_token_output_ids])
                batch_token_input_ids, batch_segment_ids,batch_labels,batch_token_output_ids = [], [],[],[]



strategy = tf.distribute.MirroredStrategy()  # 建立单机多卡策略
# with strategy.scope():  # 调用该策略
bert = build_transformer_model(
        config_path,
        checkpoint_path=None,  # 此时可以不加载预训练权重
        with_pool = True,
        with_nsp=True,
        application='unilm',
        return_keras_model=False
    )

model = bert.model  # 这个才是keras模型
# print(f'>>>> model:{model}, inputs:{model.inputs}, outputs:{model.outputs}')
print('0000000000000000000000000000000')
# print( "model:",model, "inputs:",model.inputs, "outputs:",model.outputs)
encoder = keras.models.Model(model.inputs, model.outputs[0])
seq2seq = keras.models.Model(model.inputs, model.outputs[1])
# outputs = TotalLoss([2, 3])(model.inputs + model.outputs)
# print(f'>>>> outputs:{outputs}') # 这行日志打不出来
model = keras.models.Model(model.inputs, model.outputs)
AdamW = extend_with_weight_decay(Adam, 'AdamW')
optimizer = AdamW(learning_rate=2e-5, weight_decay_rate=0.01)
model.compile(optimizer=optimizer,\
            loss={'NSP-Proba':'sparse_categorical_crossentropy','MLM-Activation':'sparse_categorical_crossentropy'},\
            loss_weights={'NSP-Proba':0.2,'MLM-Activation':10})
model.summary()
bert.load_weights_from_checkpoint(checkpoint_path) #必须最后才加载预训练权重

class AutoTitle(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        self.src_lang = np.argmax(model.predict([token_ids, segment_ids])[0])
        return self.last_token(model).predict([token_ids, segment_ids])[1]

    def generate(self, text, topk=1):
        max_c_len = maxlen - self.maxlen
        token_ids, segment_ids = tokenizer.encode(text, maxlen=max_c_len)
        # src = model.predict([token_ids, segment_ids])[0]
        # print('2222222222222222222222222')
        # print(src)
        output_ids = self.beam_search([token_ids, segment_ids],topk=topk)  # 基于beam search
        # print('000000000000000000000')
        # print(self.src_lang)
        return tokenizer.decode(output_ids),self.src_lang


autotitle = AutoTitle(start_id=None, end_id=tokenizer._token_end_id, maxlen=64)


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.lowest = 1e10

    def on_epoch_end(self, epoch, logs=None):
        # 保存最优
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            model.save_weights('./iwslt2017_deen_bi_model/best_model_bi.weights')

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
#else:

#     model.load_weights('./iwslt2017_deen_bi_model/best_model_bi.weights')
# '''''''''''''''''''''test'''''''''''''''''
model.load_weights('./iwslt2017_deen_bi_model/best_model_bi.weights')
#enss = []
#for (de,en) in valid_data:
#    ens = autotitle.generate(de,topk=4)
#    enss.append(ens[0])
de = autotitle.generate('We need to develop a feminine discourse that not only honors but also implements mercy instead of revenge, collaboration instead of competition, inclusion instead of exclusion',topk=4)
print(de)
#import codecs
#def write_trans_result(filename):
#    with codecs.open(filename,'w') as f:
#        f.write('\n'.join(enss))
#write_trans_result('datasets/iwslt2017/ende/test2010.en.bi.trans')



#---------------------get source file-------------
#import codecs
#def get_src(filename1,filename2):
#     """加载数据
#     单条格式：(标题, 正文)
#     """
#    D = []
#    with open(filename1) as f:
#        for l in f:
#            title, content = l.strip().split('\t')
#            D.append(title)
#    with open(filename2,'w') as f1:
#        f1.write('\n'.join(D))
#
#get_src('/data/lidongxing/bert4keras-master3/examples/datasets/iwslt2017/ende/test2010_deen.tsv','/data/lidongxing/bert4keras-master3/examples/datasets/iwslt2017/ende/test2010.de.src')
#en2de BLEU:4.87
#de2en BLEU:13.23
