#Liangyz
#2024/5/30  18:08

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.metrics import roc_curve, roc_auc_score
#sklearn网站：
# https://scikit-learn.org/stable/index.html
# https://scikit-learn.org/stable/api/index.html

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

# Train model num 5 10分类变成2分类
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

# print(y_train_5[:20])

from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
print("1600号数字是:\n",y[1600])
print("1600号数字的判断结果是:\n", sgd_clf.predict([X[1600]]))

# Cross validation
cvs = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring='accuracy')
print("Sklearn分三组Cross validation的分数分别是:\n", cvs)
#[0.984 0.982 0.994]

# Cross validation by hand
kfolds = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)
for train_index, test_index in kfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_folds = X_train[test_index]
    y_test_folds = y_train_5[test_index]

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_folds)
    n_correct = sum(y_pred==y_test_folds)
    print("手动分三组Cross validation的分数分别是:\n", n_correct/len(y_pred))
# 0.978
# 0.992
# 0.984
cvp = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

# Confusion matrix
cm = confusion_matrix(y_train_5, cvp)
print("Confusion matrix混淆矩阵是:\n", cm)
"""
得到的结果是：
[[1337   9]
 [  11   143]]

格式是：
[[TN FP]
 [FN TP]]
TN: True Negative 正确分类为负类
FP: False Positive 错误分类为正类(最好情况是0)
FN: False Negative 错误分类为负类(最好情况是0)
TP: True Positive 正确分类为正类

recall = TP/(TP+FN) 召回率
precision = TP/(TP+FP) 精度
F1 = 2/(1/recall+1/precision) F1分数: 精度和召回率的调和平均数
召回和精度更低的对结果影响更大, F1给与更低的值更高的权重
"""

# Precision and recall
precision = precision_score(y_train_5, cvp)
recall = recall_score(y_train_5, cvp)
print("精度和召回率分别是:\n", precision, recall)
# 0.972972972972973 0.9411764705882353

# F1 score
f1 = f1_score(y_train_5, cvp)
print("F_1分数是:\n", f1)
# 0.94

""" 阈值 """
y_scores = sgd_clf.decision_function([X[1600]])
print("1600号数字的决策分数是:\n", y_scores)
# t = 0
# y_some_digit_pred = (y_scores > t)

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method='decision_function')
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
print("尝试的阈值个数是:\n", thresholds.shape)

# Plot precision and recall
# plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')
# plt.plot(thresholds, recalls[:-1], 'g-', label='Recall')
# plt.xlabel('Threshold')
# plt.legend(loc='center left')
# plt.ylim([0, 1])
# plt.show()


# ROC curve
"""
TPR: True Positive Rate = Recall = TP/(TP+FN)
FPR: False Positive Rate = FP/(FP+TN)
"""

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
plt.plot(fpr, tpr, linewidth=2)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
"""
理论上, ROC曲线越靠近左上角越好, 也就是说TPR越高, FPR越低越好
k是随机分类器的曲线, 也就是说, ROC曲线越远离k线越好(越靠近左上角)(总不能比随机分类器还差吧)
完美分类器的ROC曲线是一个90度拐角的折线,面积是1
越差的分类器的ROC曲线越接近k线, 面积越小,接近于0.5
"""

# ROC AUC(面积)
roc_auc = roc_auc_score(y_train_5, y_scores)
print("ROC AUC是:\n", roc_auc)