#Liangyz
#2024/5/30  18:08

import numpy as np
import os

from sklearn.datasets import load_digits

# Load data
digits = load_digits()
X = digits.data
y = digits.target

#show the data
print(X.shape)
print(y.shape)

# print(y[:20])

#randomly
shuffle_index = np.random.permutation(len(X))
X, y = X[shuffle_index], y[shuffle_index]

# Split data
X_train, X_test, y_train, y_test = X[:1500], X[1500:], y[:1500], y[1500:]

# Train model num 5
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

# print(y_train_5[:20])

from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier()
sgd_clf.fit(X_train, y_train_5)
# print(y[1600])
# print(sgd_clf.predict([X[1600]]))

# Cross validation
from sklearn.model_selection import cross_val_score
# cross_val_score(sgd_clf, X_train, y_train_5, cv=5, scoring='accuracy')
print(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring='accuracy'))

# Cross validation by hand
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

kfolds = StratifiedKFold(n_splits=3)
for train_index, test_index in kfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_folds = X_train[test_index]
    y_test_folds= y_train_5[test_index]

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_folds)
    print(sum(y_pred == y_test_folds) / len(y_test_folds))
