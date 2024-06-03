#Liangyz
#2024/6/3  20:11

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from lg_study_by_hand import LogisticRegression

# ---------------------------------------------Load data---------------------------------------------------
data = pd.read_csv('../../data/microchips-tests.csv')
# print(data)

validities = np.unique(data['validity'])
print(validities)

x_axis = data.columns[0]
y_axis = data.columns[1]

for validity in validities:
    plt.scatter(data[x_axis][data['validity'] == validity],
                data[y_axis][data['validity'] == validity],
                label=validity)
plt.xlabel(x_axis)
plt.ylabel(y_axis)
plt.legend()
plt.savefig('microchips.png')
plt.show()


num_exp = data.shape[0]
x_train = data[[x_axis, y_axis]].values.reshape((num_exp, 2))
y_train = data['validity'].values.reshape((num_exp, 1))

max_iter = 100000
poly_degree = 5
sin_degree = 0

#训练
logistic_regression = LogisticRegression(x_train, y_train, polynomial_degree=poly_degree, sinusoid_degree=sin_degree, normalize_data=False)
theta, cost_histories = logistic_regression.train(max_iter)

columns = []
for theta_index in range(0,theta.shape[1]):
    columns.append('theta' + str(theta_index))

labels = logistic_regression.unique_labels

plt.plot(range(len(cost_histories[0])), cost_histories[0], label=labels[0])
plt.plot(range(len(cost_histories[1])), cost_histories[1], label=labels[1])

plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost Function')
plt.legend()
plt.savefig('microchips_cost.png')
plt.show()

y_train_pred = logistic_regression.predict(x_train)

precision = np.sum(y_train_pred == y_train) / y_train.shape[0] * 100
print('训练集准确率: ', precision, '%')

num_exps = x_train.shape[0]
samples = 200

x_min = np.min(x_train[:, 0])
x_max = np.max(x_train[:, 0])

y_min = np.min(x_train[:, 1])
y_max = np.max(x_train[:, 1])

X = np.linspace(x_min, x_max, samples)
Y = np.linspace(y_min, y_max, samples)

Z_valid = np.zeros((samples, samples))

for x_index, x in enumerate(X):
    for y_index, y in enumerate(Y):
        data = np.array([[x, y]])
        prediction = logistic_regression.predict(data)
        Z_valid[x_index][y_index] = 1 if prediction == labels[0] else 0

postive = np.where(y_train == 1)
negative = np.where(y_train == 0)

plt.figure(figsize=(16, 8))
plt.subplot(121)
plt.scatter(x_train[postive, 0], x_train[postive, 1], marker='o', c='b', label='valid')
plt.scatter(x_train[negative, 0], x_train[negative, 1], marker='x', c='r', label='invalid')
plt.contour(X, Y, Z_valid, levels=[0.5], colors='green')
plt.subplot(122)
plt.scatter(x_train[postive, 0], x_train[postive, 1], marker='o', c='b', label='valid')
plt.scatter(x_train[negative, 0], x_train[negative, 1], marker='x', c='r', label='invalid')
plt.contour(X, Y, Z_valid.T, levels=[0.5], colors='green')
#关于转置的原因是因为contour函数的参数是(X,Y,Z),而Z_valid是(samples,samples)的矩阵,而X,Y是(samples,)的向量,所以需要转置

plt.xlabel(x_axis)
plt.ylabel(y_axis)
plt.legend()
plt.savefig('microchips_predict.png')
plt.show()







