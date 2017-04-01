# coding=utf-8
import numpy as np
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier

savePath = '../../data/predictions_randomForest.txt'

X_train = np.loadtxt('X_train.txt', dtype=int)
Y_ori = np.loadtxt('Y_train.txt', dtype=int)

X_test = np.loadtxt('X_test.txt', dtype=int)
labels = np.loadtxt('labels.txt', dtype='str')

print list(labels)

Y_train = []
for i in range(len(Y_ori)):
    Y_train.append(np.argmax(Y_ori[i]))
# Y_train = [np.argmax(Y_train[i]) for i in range(len(Y_train))]

Y_train = np.asarray(Y_train)


print X_train
print Y_train

print len(X_train)
print len(Y_train)

print len(X_train[0])

clf = OneVsRestClassifier(RandomForestClassifier(random_state=0)).fit(X_train, Y_train)
#OneVsRestClassifier(RandomForestClassifier(random_state=0))
#OneVsOneClassifier(RandomForestClassifier(random_state=0))
#OutputCodeClassifier(RandomForestClassifier(random_state=0),code_size=2, random_state=0)

print '\ntrain acc:',np.sum(np.asarray(clf.predict(X_train) == Y_train))/ (1.0 * len(Y_train))

pred = clf.predict(X_test)

fout = open(savePath,'w')
labels = list(labels)
for p in pred:
    fout.write(str(labels[p])+'\n')
fout.close()























