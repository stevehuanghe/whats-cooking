# coding=utf-8
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from data_factory import DataManager
import numpy as np
import pandas as pd
trainParh = '../data/train.json'
testPath = '../data/test.json'
savePath = '../data/predictions_nn.txt'
dm = DataManager()

train_data_json, test_data_json = dm.load_data(trainParh,testPath)
train_data, train_label, test_data, vocab, labels = dm.process_data(train_data_json,test_data_json,tf_idf=False)

model = Sequential()
model.add(Dense(1000, activation='relu', input_dim=7137))
model.add(Dropout(0.5))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(20, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
                           optimizer=sgd,
                           metrics=['accuracy'])
model.fit(train_data, train_label,epochs = 300, batch_size=2048)

score = model.evaluate(train_data, train_label, batch_size=1024)
print '\ntrain acc:',score
print model.metrics_names
pred = model.predict(test_data, batch_size=512)


fout = open(savePath,'w')
labels = list(labels)
for i in range(len(test_data_json)):
    id = test_data_json[i][u'id']
    c = np.argmax(pred[i])
    fout.write(str(id) + '\t' + str(labels[c])+'\n')
fout.close()






















