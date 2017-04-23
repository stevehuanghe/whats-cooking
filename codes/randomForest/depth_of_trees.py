# coding=utf-8
import numpy as np
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import metrics

savePath = './results/predictions_randomForest.txt'

#x = np.loadtxt('x_train', dtype=int)
#y_ori = np.loadtxt('y_train', dtype=int)
x = np.loadtxt('X_train.txt', dtype=int)
y_ori = np.loadtxt('Y_train.txt', dtype=int)

labels = np.loadtxt('labels.txt', dtype='str')
ids = np.loadtxt('ids.txt', dtype='str')

y = []
for i in range(len(y_ori)):
    y.append(np.argmax(y_ori[i]))

y = np.asarray(y)
print len(y)

# train set data split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# tuning max features
num_estimator = 200
m_feature = 'log2'
msl = 1
max_depths = [100, 150, 200, 250, 300, 350, 400, 450, 500]

for md in max_depths :
    model = RandomForestClassifier(n_estimators = num_estimator, n_jobs = -1,random_state = 50,max_features = m_feature, min_samples_leaf=msl, max_depth = md)
    model.fit(x_train, y_train)
    #results = cross_val_predict(model, x_train, y_train, cv=2)
    #print metrics.accuracy_score(y_train, results)
    y_pred = model.predict(x_test)
    #print(metrics.classification_report(y_test, y_pred))
    print(metrics.accuracy_score(y_test, y_pred))

