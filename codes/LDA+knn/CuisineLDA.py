from __future__ import division
from Data_Factory_main import DataManager
import numpy as np
import lda
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def __build_vocab(train_data, test_data):
    vocab = set()
    train_num = len(train_data)
    test_num = len(test_data)
    for ingrd in train_data:
        for i in ingrd:
           vocab.add(i)
    for ingrd in test_data:
        for i in ingrd:
           vocab.add(i)
    vocab = list(vocab)
    sorted(vocab)
    print 'size of vocabulary:', len(vocab)
    return vocab


def __to_BoW(data,vocab):
    num_data = len(data)
    num_vocab = len(vocab)
    data_vec = []
    for i in range(num_data):
        vec = np.zeros(num_vocab, dtype=int)
        items = data[i]
        for n,item in enumerate(items):
            vec[vocab.index(item)] = n
        data_vec.append(vec)
    return data_vec



def main():
    trainPath = 'data/train.json'
    testPath = 'data/test.json'
    train_data_path = 'data/train_data.npy'
    train_label_path = 'data/train_label.npy'
    test_data_path = 'data/test_data.npy'
    test_label_path = 'data/test_label.npy'
    label_set_path = 'data/label_set.txt'
    train_doc_path='data/train_doc_set.txt'
    test_doc_path = 'data/test_doc_set.txt'
    vocabulary_path = 'data/vocabulary_set.txt'

    dm = DataManager()

    train_data_vec, train_label_vec, train_doc, test_data_vec,test_label_vec, test_doc, label_set, vocab = dm.load_processed_data( train_data_path, train_label_path,train_doc_path,
                                                                                                                    test_data_path, test_label_path,test_doc_path,
                                                                                                                    vocabulary_path,label_set_path)

    # train_data_json, test_data_json = dm.load_data(trainParh, testPath)

    train={}
    test={}
    test_topic={}
    print("type(train_data_vec): {}".format(type(train_data_vec)))
    train_data=np.asarray(train_data_vec)
    train_data.astype(int)
    test_data=np.asarray(test_data_vec)
    test_data.astype(int)
    # for i in train_data:
    #     for j in i:
    #        print j
    print train_data
    # print test_data
    model_train = lda.LDA(n_topics=20, n_iter=500, random_state=1)
    model_train.fit(train_data)
    doc_topic = model_train.doc_topic_
    n = 20
    for i, topic_dist in enumerate(model_train.topic_word_): ##get the topic_number + key words
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n + 1):-1]
        topic_words=topic_words.tolist()
        train[i]=topic_words


    model_test = lda.LDA(n_topics=20, n_iter=500, random_state=1)
    model_test.fit(test_data)
    doc_topic = model_test.doc_topic_
    # topic_word_test = model_test.topic_word_
    n = 20
    for i, topic_dist in enumerate(model_test.topic_word_):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n + 1):-1]
        topic_words = topic_words.tolist()
        test[i]=topic_words
    for n, topic_dist in enumerate(doc_topic):  ##get the document + topic_number  test
        topic_most_pr = doc_topic[n].argmax()
        test_topic[n]=topic_most_pr

    vocab=__build_vocab(train.values(),test.values())
    train_vec=__to_BoW(train.values(), vocab)
    test_vec=__to_BoW(test.values(), vocab)
    neigh = KNeighborsClassifier(n_neighbors=1)
    # print train_label_vec
    neigh.fit(train_vec, train.keys())
    y_ped=neigh.predict(test_vec)
    print(neigh.predict(test_vec))
    test_to_train={}
    j=0
    for i in y_ped:
        test_to_train[j]=i
        j=j+1
    print test_to_train

    y_test=test_topic.values()
    y_pred=[]
    for i in y_test:
        y_pred.append(test_to_train[i])
    y_true=[]
    for i in test_label_vec:
        i=i.tolist()
        y_true.append(i.index(1))
    print(accuracy_score(y_true, y_pred))


    # try:
    #     plt.style.use('ggplot')
    # except:
    #     # version of matplotlib might not be recent
    #     pass
    # Vlda=VisualizationLDA()
    # plt.figure(1)
    # Vlda.topic_word_visual(topic_word)
    # plt.figure(2)
    # Vlda.doc_topic_visual(doc_topic)


if __name__ == '__main__':
    main()