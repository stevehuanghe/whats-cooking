# -*- coding: utf-8 -*-


from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from data_factory import DataManager 
import numpy as np
trainPath ='.../data/train.json'
testPath = '.../data/test.json'
savePath = '.../data/predictions_nb.txt'

dm = DataManager()

train_data, test_data = dm.load_data(trainPath,testPath)
train_data, train_label, test_data, vocab, labels = dm.process_data(train_data,test_data,tf_idf=False)


tLabel=[]
for i in range(len(train_label)):
     tLabel.append(np.argmax(train_label[i]))
       


print('build model...')
clf = MultinomialNB()
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)

print('model accuracy...')
scores = cross_val_score(clf, train_data,tLabel, cv=5)
print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

print('predict...')
clf.fit(train_data,tLabel)
pred=clf.predict(test_data)

labels = list(labels)

fout = open(savePath,'w')
for p in pred:
    
    fout.write(str(labels[p])+'\n')
fout.close()

