##! /bin/env python
# -*- coding: utf-8 -*-
"""
训练网络，并保存模型，其中LSTM的实现采用Python中的keras库
"""
import sys
import json
from pathlib import Path

import multiprocessing
import pandas as pd 
import numpy as np 
import jieba
import keras

from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.models import Model, model_from_json
from keras.layers import LSTM, GRU, RNN, Bidirectional, SimpleRNN
from keras.layers import GRUCell, LSTMCell, SimpleRNNCell
from keras.layers import Input, Embedding, Dense, Dropout, Activation
from keras.callbacks import EarlyStopping
from keras import metrics

from utils import AttentionLayer, SelfAttentionLayer

"""
from keras.src.layers.rnn.bidirectional import Bidirectional
from keras.src.layers.rnn.conv_lstm1d import ConvLSTM1D
from keras.src.layers.rnn.conv_lstm2d import ConvLSTM2D
from keras.src.layers.rnn.conv_lstm3d import ConvLSTM3D
from keras.src.layers.rnn.gru import GRU
from keras.src.layers.rnn.gru import GRUCell
from keras.src.layers.rnn.lstm import LSTM
from keras.src.layers.rnn.lstm import LSTMCell
from keras.src.layers.rnn.rnn import RNN
from keras.src.layers.rnn.simple_rnn import SimpleRNN
from keras.src.layers.rnn.simple_rnn import SimpleRNNCell
from keras.src.layers.rnn.stacked_rnn_cells import StackedRNNCells
from keras.src.layers.rnn.time_distributed import TimeDistributed
"""

np.random.seed(1337)  # For Reproducibility
sys.setrecursionlimit(1000000)

# set parameters:
cpu_count = multiprocessing.cpu_count() # 4
vocab_dim = 100
n_iterations = 1  # ideally more..
n_exposures = 10 # 所有频数超过10的词语
window_size = 7
input_length = 100
maxlen = 100

n_epoch = 10
batch_size = 32


def loadfile(data_path):
    if not isinstance(data_path, Path):
        data_path = Path(data_path)
    neg=pd.read_csv(data_path / 'neg.csv', header=None, index_col=None)
    pos=pd.read_csv(data_path / 'pos.csv', header=None, index_col=None, on_bad_lines='skip')
    neu=pd.read_csv(data_path / 'neutral.csv', header=None, index_col=None)

    pos.fillna('', inplace=True)
    neu.fillna('', inplace=True)
    neg.fillna('', inplace=True)

    combined = np.concatenate((pos[0], neu[0], neg[0]))
    y = np.concatenate((np.ones(len(pos), dtype=int),
                        np.zeros(len(neu), dtype=int), 
                        -1*np.ones(len(neg),dtype=int)))

    return combined, y


#对句子经行分词，并去掉换行符
def tokenizer(text):
    ''' Simple Parser converting each document to lower-case, then
        removing the breaks for new lines and finally splitting on the
        whitespace
    '''
    text = [jieba.lcut(document.replace('\n', '')) for document in text]
    return text


def create_dictionaries(model=None,
                        combined=None):
    ''' Function does are number of Jobs:
        1- Creates a word to index mapping
        2- Creates a word to vector mapping
        3- Transforms the Training and Testing Dictionaries

    '''
    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.key_to_index.keys(),
                            allow_update=True)
        #  freqxiao10->0 所以k+1
        w2indx = {v: k+1 for k, v in gensim_dict.items()}#所有频数超过10的词语的索引,(k->v)=>(v->k)
        w2vec = {word: model.wv[word] for word in w2indx.keys()}#所有频数超过10的词语的词向量, (word->model(word))

        def parse_dataset(combined): # 闭包-->临时使用
            ''' Words become integers
            '''
            data=[]
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0) # freqxiao10->0
                data.append(new_txt)
            return data # word=>index
        combined = parse_dataset(combined)
        combined = sequence.pad_sequences(combined, maxlen=maxlen)#每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
        return w2indx, w2vec, combined
    else:
        print('No data provided...')


#创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def word2vec_train(combined, dir_path, model_name):
    if not isinstance(dir_path, Path):
        dir_path = Path(dir_path)
    model_path = dir_path / model_name
    if not model_path.exists():
        model = Word2Vec(vector_size=vocab_dim,
                        min_count=n_exposures,
                        window=window_size,
                        workers=cpu_count,
                        epochs=n_iterations)
        model.build_vocab(combined) # input: list
        model.train(combined, total_examples=model.corpus_count, epochs=n_iterations)
        
        if not dir_path.exists():
            dir_path.mkdir()
        model.save(str(model_path))
    else:
        model = Word2Vec.load(str(model_path))
    index_dict, word_vectors, combined=create_dictionaries(model=model,combined=combined)
    return index_dict, word_vectors, combined


def get_data(index_dict,word_vectors,combined,y):
    """数据集划分"""
    n_symbols = len(index_dict) + 1  # 所有单词的索引数，频数小于10的词语索引为0，所以加1
    embedding_weights = np.zeros((n_symbols, vocab_dim)) # 初始化 索引为0的词语，词向量全为0
    for word, index in index_dict.items(): # 从索引为1的词语开始，对每个词语对应其词向量
        embedding_weights[index, :] = word_vectors[word]
    x_train, x_test, y_train, y_test = train_test_split(combined, y, test_size=0.2)
    y_train = keras.utils.to_categorical(y_train, num_classes=3) 
    y_test = keras.utils.to_categorical(y_test, num_classes=3)
    # print x_train.shape,y_train.shape
    return n_symbols, embedding_weights, x_train, y_train, x_test, y_test


##定义网络结构
def train_model(n_symbols,embedding_weights,x_train,y_train, x_test, y_test, model_name, model_save_path, attention=None):
    if not isinstance(model_save_path, Path):
        model_save_path = Path(model_save_path)

    print('Defining a Simple Keras Model...')
    
    inputs = Input(shape=(input_length, )) # 输入层

    embedding_layer = Embedding(output_dim=vocab_dim,
                        input_dim=n_symbols,
                        mask_zero=True,
                        weights=[embedding_weights])(inputs)  # Adding Input Length
    # LSTM, GRU, RNN, ConvLSTM1D, Bidirectional, SimpleRNN
    units = 50
    activation = 'tanh'
    if model_name.startswith("rnn"):
        cell_name = model_name.split('_')[1]
        # GRUCell, LSTMCell, SimpleRNNCell, StackedRNNCells
        if cell_name=='simple':
            cell = SimpleRNNCell(units, activation=activation)
        elif cell_name=='gru':
            cell = GRUCell(units, activation=activation)
        elif cell_name=='lstm':
            cell = LSTMCell(units, activation=activation)
        else:
            raise KeyError(f"Wrong cell_name with {cell_name}!")
        backbone = RNN(cell)(embedding_layer)
    elif model_name=="simple_rnn":
        backbone = SimpleRNN(units, activation=activation)(embedding_layer)
    elif model_name=='gru':
        backbone = GRU(units, activation=activation)(embedding_layer)
    elif model_name.startswith('bidirectional'):
        cell_name = model_name.split('_')[1]
        if cell_name=='lstm':
            backbone, forward_h, forward_c, backward_h, backward_c = Bidirectional(LSTM(units, activation=activation, return_sequences=True, return_state=True))(embedding_layer)

        elif cell_name=='gru':
            backbone = Bidirectional(GRU(units, activation=activation, return_sequences=True))(embedding_layer)
    elif model_name=='lstm':
        backbone = LSTM(units, activation=activation)(embedding_layer)
    else:
        raise KeyError(f"Wrong model_name with {model_name}!")
    if attention is not None:
        if attention=='attention':
            # attention
            attention_layer = AttentionLayer()(backbone[:, -1, :])
        elif attention=='self-attention':
            # self-attention
            attention_layer = SelfAttentionLayer()(backbone[:, -1, :])
        else:
            raise KeyError(f"Wrong attention with {attention}")
        dropout_layer = Dropout(0.5)(attention_layer)
    else:
        dropout_layer = Dropout(0.5)(backbone[:, -1, :])
    dense_layer = Dense(3, activation='softmax')(dropout_layer) # Dense=>全连接层,输出维度=3
    # activation_layer = Activation('softmax')(dense_layer)

    model = Model(inputs=inputs, outputs=dense_layer)

    early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    print('Compiling the Model...')
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',metrics=['accuracy', metrics.Precision(), metrics.Recall(), metrics.F1Score(average='macro')])

    print("Train...") # batch_size=32

    # 创建训练集和验证集
    train_size = int(0.9 * len(x_train))
    x_train, x_val = x_train[:train_size], x_train[train_size:]
    y_train, y_val = y_train[:train_size], y_train[train_size:]

    model.fit(x_train, y_train, batch_size=batch_size,  
              epochs=n_epoch, verbose=1, 
              callbacks=[early_stopping],
              validation_data=(x_val, y_val))

    print("Evaluate...")
    loss, accuracy, precision, avg_recall, f1_score = model.evaluate(x_test, y_test,
                    batch_size=batch_size)
    print(f'Test results:\n{loss=}, {accuracy=}, {precision=}, {avg_recall=}, {f1_score=}')

    json_string = model.to_json()
    with open(model_save_path / f'{model_name}.json', 'w') as outfile:
        outfile.write( json.dumps(json_string) )
    model.save_weights(model_save_path / f'{model_name}.weights.h5')
    print(f"Model saved in {model_save_path}/{model_name}.weights.h5")

if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_path", type=str, default="./data", help="data directory")
    parser.add_argument("-m", "--model_name", type=str, default="lstm", help="set model to use")
    parser.add_argument("-s", "--save_path", type=str, default="./model", help="where to save model")
    parser.add_argument("-a", "--attention", type=str, default=None, help="use which attention or not to use")
    args = parser.parse_args()

    model_name = args.model_name
    model_save_path = args.save_path
    data_path = args.data_path
    attention = args.attention
    #训练模型，并保存
    print('Loading Data...')
    combined, y=loadfile(data_path)
    print(len(combined), len(y))
    print('Tokenising...')
    combined = tokenizer(combined)
    print('Training a Word2vec model...')
    index_dict, word_vectors, combined = word2vec_train(combined, './word2vec', 'Word2vec_model.pkl')

    print('Setting up Arrays for Keras Embedding Layer...')
    n_symbols, embedding_weights, x_train, y_train, x_test, y_test = get_data(index_dict, word_vectors,combined,y)
    print("x_train.shape and y_train.shape:")
    print(x_train.shape, y_train.shape)
    train_model(n_symbols, embedding_weights, x_train, y_train, x_test, y_test, model_name, model_save_path)
