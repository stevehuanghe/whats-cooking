from __future__ import division
from data_factory import DataManager
import numpy as np
import lda
import matplotlib.pyplot as plt

class ClusteringLDA(object):
    def topic_word(self, model, vocabulary):
        topic_word = model.topic_word_
        print "type(topic_word): {}".format(type(topic_word))
        print "shape: {}".format(topic_word.shape)
        n = 5
        for i, topic_dist in enumerate(topic_word):
            topic_words = np.array(vocabulary)[np.argsort(topic_dist)][:-(n + 1):-1]
            print '*Topic {}\n- {}'.format(i, '   '.join(topic_words))
        return topic_word

    def Document_topic(self, model, test_doc):
        doc_topic = model.doc_topic_
        print "type(doc_topic): {}".format(type(doc_topic))
        print "shape: {}".format(doc_topic.shape)
        for n in range(10):
            topic_most_pr = doc_topic[n].argmax()
            print("doc: {} topic: {}\n{}...".format(n, topic_most_pr, test_doc[n][:50]))
        return doc_topic

class VisualizationLDA(object):
    def topic_word_visual(self,topic_word):
        f, ax = plt.subplots(5, 1, figsize=(8, 6), sharex=True)
        for i, k in enumerate([0, 5, 9, 14, 19]):
            ax[i].stem(topic_word[k, :], linefmt='b-',
                       markerfmt='bo', basefmt='w-')
            ax[i].set_xlim(-50, 4500)
            ax[i].set_ylim(0, 0.139)
            ax[i].set_ylabel("Prob")
            ax[i].set_title("topic {}".format(k))

        ax[4].set_xlabel("word")

        plt.tight_layout()
        plt.show()

    def doc_topic_visual(self,doc_topic):
        f, ax = plt.subplots(5, 1, figsize=(8, 6), sharex=True)
        for i, k in enumerate([1, 3, 4, 8, 9]):
            ax[i].stem(doc_topic[k, :], linefmt='r-',
                       markerfmt='ro', basefmt='w-')
            ax[i].set_xlim(-1, 21)
            ax[i].set_ylim(0, 1.1)
            ax[i].set_ylabel("Prob")
            ax[i].set_title("Document {}".format(k))

        ax[4].set_xlabel("Topic")

        plt.tight_layout()
        plt.show()


def main():
    # trainPath = 'train.json'
    testPath = 'test.json'
    dm = DataManager()
    test_data = dm.load_data(testPath)
    test_data_vec, vocabulary,test_doc = dm.process_data(test_data)
    print("type(test_data_vec): {}".format(type(test_data_vec)))
    test_data_vec=np.asarray(test_data_vec)
    model = lda.LDA(n_topics=20, n_iter=500, random_state=1)
    model.fit(test_data_vec)
    Clda=ClusteringLDA()
    topic_word=Clda.topic_word(model, vocabulary)
    doc_topic = Clda.Document_topic(model,test_doc)
    try:
        plt.style.use('ggplot')
    except:
        # version of matplotlib might not be recent
        pass
    Vlda=VisualizationLDA()
    plt.figure(1)
    Vlda.topic_word_visual(topic_word)
    # plt.figure(2)
    # Vlda.doc_topic_visual(doc_topic)


if __name__ == '__main__':
    main()