#! /bin/env python
# -*- coding: utf-8 -*-
"""
预测
"""
import sys
import json
import jieba
import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from keras.preprocessing import sequence
from keras.models import model_from_json

np.random.seed(1337)  # For Reproducibility
sys.setrecursionlimit(1000000)
# define parameters
maxlen = 100

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
        return w2indx, w2vec,combined
    else:
        print('No data provided...')


def input_transform(string):
    words = jieba.lcut(string)
    words = np.array(words).reshape(1,-1)
    model = Word2Vec.load('../word2vec/Word2vec_model.pkl')
    _, _, combined = create_dictionaries(model, words)
    return combined


def other_predict(string, model_name):
    print('loading model......')
    with open(f'./model/{model_name}.json', 'r') as f:
        json_string = json.load(f)
    model = model_from_json(json_string)

    print(f'loading {model_name} weights......')
    model.load_weights(f'./model/{model_name}.weights.h5')
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    data = input_transform(string)
    data.reshape(1, -1)
    #print data
    result = model.predict(data)
    result = np.argmax(result, axis=1)
    # print result # [[1]]
    if result[0]==1:
        print(string,' positive')
    elif result[0]==0:
        print(string,' neural')
    else:
        print(string,' negative')


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-msg", "--message", type=str)
    parser.add_argument("-m", "--model", type=str, default='lstm')
    args = parser.parse_args()

    string = args.message
    model_name = args.model
    other_predict(string, model_name)
