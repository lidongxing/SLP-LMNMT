#! -*- coding: utf-8 -*-
# take bert for NMT task and employ the UNILM seq2seq method
#refer to bert4keras：https://github.com/bojone/bert4keras
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
from bert4keras.optimizers import Adam,is_tf_keras
from bert4keras.snippets import sequence_padding, open
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from bert4keras.optimizers import extend_with_weight_decay
from rouge import Rouge  # pip install rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import random as rand
import tensorflow as tf
#from keras.utils import plot_model
from keras import losses


# hpyer-parameters
maxlen = 128
batch_size = 32
epochs = 8

# bert config
#multi-language BERT pretrained model
config_path = '/models/multi_cased_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/models/multi_cased_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/models/multi_cased_L-12_H-768_A-12/vocab.txt'



def load_data(filename):
    """load data
    item：(it, en)
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            title, content = l.strip().split('\t')
            D.append((title, content))
    return D

import random
# load datasets
train_data = load_data('datasets/iwslt2017/corpus_iwslt2017.tsv')
random.shuffle(train_data)
valid_data = load_data('data/ennl/dev2010_nlen.tsv')


# load dict and tokenize
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
# tokenizer = Tokenizer(token_dict, do_lower_case=True)
# build tokenization
tokenizer = Tokenizer(dict_path, do_lower_case=True)

class data_generator(DataGenerator):
    """data generator
    """
    def __iter__(self, random=False):
        batch_token_input_ids, batch_segment_ids,batch_labels,batch_token_output_ids = [], [],[],[]
        for is_end, (title_, content) in self.sample(random):
            output = title_.split(' ',1)
            label,title = output[0],output[1]
            token_ids, segment_ids = tokenizer.encode(
                title, content, maxlen=maxlen
            )
            label = float(label)
            # print(label)

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



#strategy = tf.distribute.MirroredStrategy()
# with strategy.scope():
bert = build_transformer_model(
        config_path,
        checkpoint_path=None, 
        with_pool = True,
        with_nsp=True,
        application='unilm',
        return_keras_model=False
    )

model = bert.model
# print(f'>>>> model:{model}, inputs:{model.inputs}, outputs:{model.outputs}')
# print( "model:",model, "inputs:",model.inputs, "outputs:",model.outputs)
encoder = keras.models.Model(model.inputs, model.outputs[0])
seq2seq = keras.models.Model(model.inputs, model.outputs[1])
# outputs = TotalLoss([2, 3])(model.inputs + model.outputs)
# print(f'>>>> outputs:{outputs}') 
label_output = keras.layers.Lambda(lambda x: x[0], name='label-token')(model.outputs)
label_output_findall= keras.layers.Dense(
    units=4,
    activation='softmax',
    kernel_initializer=bert.initializer,
    name='label_output'
)(label_output)
# output_labels1 = keras.layers.Dense(16,activation='softmax',name='output_labels1')(model.outputs[0])
# output_labels = keras.layers.Dense(2,activation='softmax',name='output_labels')(model.outputs[0])

model = keras.models.Model(model.inputs, [label_output_findall,model.outputs[1]])
AdamW = extend_with_weight_decay(Adam, 'AdamW')
optimizer = AdamW(learning_rate=1e-5, weight_decay_rate=0.01)
# adam = tf.keras.optimizers.Adam(lr=2e-5, clipnorm=1)
model.compile(optimizer=optimizer,\
            loss={'label_output':'sparse_categorical_crossentropy','MLM-Activation':'sparse_categorical_crossentropy'},\
            loss_weights={'label_output':0.1,'MLM-Activation':10})
model.summary()
# plot_model(model,to_file='slp_model.png',show_shapes=True)
bert.load_weights_from_checkpoint(checkpoint_path)

class AutoTitle(AutoRegressiveDecoder):
    """seq2seq decoder
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
        # print(src)
        output_ids = self.beam_search([token_ids, segment_ids],topk=topk)  # topk=beam size
        # print(self.src_lang)
        return tokenizer.decode(output_ids),self.src_lang


autotitle = AutoTitle(start_id=None, end_id=tokenizer._token_end_id, maxlen=64)


class Evaluator(keras.callbacks.Callback):
    """evaluate and save the model
    """
    def __init__(self):
        self.lowest = 1e10

    def on_epoch_end(self, epoch, logs=None):
        # save the best model
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            model.save_weights('./iwslt2017_many2en_models/best_model_many2en.weights')

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
else:
    model.load_weights('./iwslt2017_many2en_models/best_model_many2en.weights')
# '''''''''''''''''''''test'''''''''''''''''
model.load_weights('./iwslt2017_many2en_models/best_model_many2en.weights')
enss = []
for (es,en) in valid_data:
    ens = autotitle.generate(es,topk=4)
    enss.append(ens[0])
import codecs
def write_trans_result(filename):
    with codecs.open(filename,'w') as f:
        f.write('\n'.join(enss))
write_trans_result('datasets/iwslt2017/ennl/dev2010.manytoen.en.trans2')


#---------------------get source file-------------
import codecs
def get_src(filename1,filename2):
    D = []
    with open(filename1) as f:
        for l in f:
            title, content = l.strip().split('\t')
            D.append(title)
    with open(filename2,'w') as f1:
        f1.write('\n'.join(D))
get_src('datasets/iwslt2017/ennl/test2010_nlen.tsv','datasets/iwslt2017/ennl/test2010_many2nl.nl.src')
