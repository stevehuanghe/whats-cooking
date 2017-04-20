# coding: utf-8
# He Huang
# 3/14/17

import os
import numpy as np
import pickle as pk
import json
from sklearn.feature_extraction.text import TfidfVectorizer,TfidfTransformer

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
    input:
        - train_data: training data in json 
        - test_data: testing data in json
        - tf_idf: bool type, set True to use tf-idf, set False to use BoW
    output:
        - train_data_vec: training data, bag-of-words vector
        - train_label_vec: training data's labels, one-hot vector
        - test_data_vec: testing data, bag-of-words vector
        - vocab: vocabulary, a list of all ingredients
        - label_set: a list of all possilbe labels (20)
    '''
    def process_data(self,train_data,test_data,tf_idf = True):
        print 'building vocabulary...'
        vocab = self.__build_vocab(train_data,test_data)
        print 'building label set...'
        label_set = self.__build_labels(train_data)

        print 'processing train data...'
        train_data_vec = self.__to_BoW(train_data,u'ingredients',vocab)
        print 'processing test data...'
        test_data_vec = self.__to_BoW(test_data,u'ingredients',vocab)
        if tf_idf == True:
            print 'calculating tf-idf...'
            train_data_vec,test_data_vec = self.tf_idf_from_BoW(train_data_vec,test_data_vec)
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
        label_set = sorted(label_set)
        print 'size of label set:',len(label_set)
        return label_set
    
    def __to_BoW(self,data,key,vocab):
        num_data = len(data)
        num_vocab = len(vocab)
        data_vec = np.zeros((num_data,num_vocab),dtype=int)
        for i in range(num_data):
            items = data[i][key]
            if type(items) != type(list()):
                data_vec[i,vocab.index(items)] = 1
            else:
                for item in items:
                    data_vec[i,vocab.index(item)] = 1
        return data_vec
    
    def tf_idf_from_BoW(self,train_data,test_data):
        transformer = TfidfTransformer()
        transformer.fit(train_data)
        return transformer.transform(train_data).todense(),transformer.transform(test_data).todense()
    
    def split_train_test(self,data,labels,label_set,p_train):
        num_label_set = len(label_set)
        num_data = data.shape[0]
        label_list = []
        for i in range(num_label_set):
            label_list.append([])
        label_prob = np.zeros(num_label_set)
        idx = 0
        for label in labels:
            c = np.argmax(label)
            label_list[c].append(idx)
            idx += 1
            label_prob[c] += 1.0
        label_prob = label_prob / len(labels)
        print label_prob
        num_train = int(num_data * p_train)
        num_test = num_data - num_train
        train_data = np.ndarray((num_train,data.shape[1]),float)
        test_data = np.ndarray((num_test,data.shape[1]),float)
        train_label = np.ndarray((num_train,labels.shape[1]),int)
        test_label = np.ndarray((num_test,labels.shape[1]),int)
        train_idx = []
        test_idx = []
        num_each = (label_prob*num_test).astype(int)
        num_each[-1] = num_test - sum(num_each[:-1])
        print num_each
        print num_test
        for i in range(num_label_set):
            idxs = np.random.choice(len(label_list[i]),num_each[i],replace=False)
            for idx in idxs:
                test_idx.append(int(idx))
        for i in range(num_data):
            if i not in test_idx:
                train_idx.append(i)

        train_idx = np.random.permutation(train_idx)
        test_idx = np.random.permutation(test_idx)

        for i in range(num_train):
            idx = train_idx[i]
            train_data[i] = data[idx]
            train_label[i] = labels[idx]
        for i in range(num_test):
            idx = test_idx[i]
            test_data[i] = data[idx]
            test_label[i] = labels[idx]

        return train_data,train_label,test_data,test_label
        
    def load_processed_data(self,train_data_path,train_label_path,test_data_path,test_label_path,label_set_path):
        train_data = np.load(train_data_path)
        train_label = np.load(train_label_path)
        test_data = np.load(test_data_path)
        test_label = np.load(test_label_path)
        with open(label_set_path,'rb') as fin:
            label_set = pk.load(fin)
        return train_data,train_label,test_data,test_label,label_set

def main():
    trainPath = '../data/train.json'
    testPath = '../data/test.json'
    train_data_path = '../data/train_data'
    train_label_path = '../data/train_label'
    test_data_path = '../data/test_data'
    test_label_path = '../data/test_label'
    label_set_path = '../data/label_set.txt'
    dm = DataManager()
    train_data, test_data = dm.load_data(trainPath,testPath)
    train_data_vec, train_label_vec, test_data_vec, vocabulary, label_set = dm.process_data(train_data,test_data,tf_idf=False)
    train_data_vec,_ = dm.tf_idf_from_BoW(train_data_vec,test_data_vec)
    print train_data_vec.shape
    train_data,train_label,test_data, test_label = dm.split_train_test(train_data_vec,train_label_vec,label_set,0.9)
    np.save(train_data_path,train_data)
    np.save(train_label_path,train_label)
    np.save(test_data_path,test_data)
    np.save(test_label_path,test_label)
    with open(label_set_path,'wb') as fout:
        pk.dump(label_set,fout)
    
    print 'finished.'

if __name__ == '__main__':
    main()


