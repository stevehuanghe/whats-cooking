# coding=utf-8
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from data_factory import DataManager
import numpy as np
trainParh = '../data/train.json'
testPath = '../data/test.json'
savePath = '../data/predictions_nn.txt'
dm = DataManager()

train_data, test_data = dm.load_data(trainParh,testPath)
train_data, train_label, test_data, vocab, labels = dm.process_data(train_data,test_data)
#train_label = [np.where(label==1)[0] for label in train_label]

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(64, activation='relu', input_dim=7137))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(20, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
                           optimizer=sgd,
                           metrics=['accuracy'])
#model.fit(train_data,train_label)
model.fit(train_data, train_label,epochs = 100, batch_size=512)

score = model.evaluate(train_data, train_label, batch_size=512)
print '\ntrain acc:',score
print model.metrics_names
pred = model.predict(test_data, batch_size=512)

fout = open(savePath,'w')
labels = list(labels)
for p in pred:
    c = np.argmax(p)
    fout.write(str(labels[c])+'\n')
fout.close()























