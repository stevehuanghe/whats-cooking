# coding: utf-8
# He Huang
# 3/14/17

import os
import numpy as np
import json

class DataManager(object):
    '''
    read data and return json object
    '''
    def load_data(self, train_file, test_file):
        train_data = self.__read_data(train_file)
        test_data = self.__read_data(test_file)
        train_num = len(train_data)
        test_num = len(test_data)
        print 'data loaded'
        print 'num train:',train_num,'; num test:',test_num
        return train_data,test_data
    
    '''
    turn data into Bag-of-Words vectors and labels into one-hot vectors
    '''
    def process_data(self,train_data,test_data):
        print 'building vocabulary...'
        vocab = self.__build_vocab(train_data,test_data)
        print 'building label set...'
        label_set = self.__build_labels(train_data)
        print 'processing train data...'
        train_data_vec = self.__to_BoW(train_data,u'ingredients',vocab)
        print 'processing test data...'
        test_data_vec = self.__to_BoW(test_data,u'ingredients',vocab)
        print 'processing labels...'
        train_label_vec = self.__to_BoW(train_data,u'cuisine',label_set)
        print 'processing done'
        return train_data_vec, train_label_vec, test_data_vec, vocab, label_set
    
    def __read_data(self,file_path):
        if not os.path.isfile(file_path):
            print 'file not found:',file_path
            exit(-1)
        data = json.loads(open(file_path).read())
        return data
    
    def __build_vocab(self,train_data,test_data):
        vocab = set()
        train_num = len(train_data)
        test_num = len(test_data)
        for i in range(train_num):
            ingredients = train_data[i][u'ingredients']
            for ingrd in ingredients:
                vocab.add(ingrd)
        for i in range(test_num):
            ingredients = test_data[i][u'ingredients']
            for ingrd in ingredients:
                vocab.add(ingrd)
        vocab = list(vocab)
        sorted(vocab)
        print 'size of vocabulary:',len(vocab)
        return vocab
    
    def __build_labels(self,train_data):
        label_set = set()
        train_num = len(train_data)
        for i in range(train_num):
            label_set.add(train_data[i][u'cuisine'])
        label_set = list(label_set)
        sorted(label_set)
        print 'size of label set:',len(label_set)
        return label_set
    
    def __to_BoW(self,data,key,vocab):
        num_data = len(data)
        num_vocab = len(vocab)
        data_vec = []
        for i in range(num_data):
            vec = np.zeros(num_vocab,dtype=int)
            items = data[i][key]
            if type(items) != type(list()):
                vec[vocab.index(items)] = 1
            else:
                for item in items:
                    vec[vocab.index(item)] = 1
            data_vec.append(vec)
        return data_vec
        
    
def main():
    trainPath = '../data/train.json'
    testPath = '../data/test.json'
    dm = DataManager()
    train_data, test_data = dm.load_data(trainPath,testPath)
    train_data_vec, train_label_vec, test_data_vec, vocabulary, label_set = dm.process_data(train_data,test_data)


if __name__ == '__main__':
    main()


