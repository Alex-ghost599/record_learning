#Liangyz
#2024/6/3  16:55

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from lg_study_by_hand import LogisticRegression

# ---------------------------------------------Load data---------------------------------------------------
iris = load_iris()
# print(iris.keys())
#'data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'
# print(iris.feature_names)
#'sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'
# print(iris.target_names)
#'setosa' 'versicolor' 'virginica'

data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['class'] = iris.target
data['class'] = data['class'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})


iris_types = iris.target_names
x_axis = iris.feature_names[2]
y_axis = iris.feature_names[3]

# for iris_type in iris_types:
#     plt.scatter(data[x_axis][data['class'] == iris_type],
#                 data[data['class'] == iris_type][y_axis],
#                 label=iris_type)
# plt.xlabel(x_axis)
# plt.ylabel(y_axis)
# plt.legend()
# plt.savefig('iris.png')
# plt.show()

num_examples = data.shape[0]
x_train = data[[x_axis, y_axis]].values.reshape((num_examples, 2))
# print(x_train.shape)
y_train = data['class'].values.reshape((num_examples, 1))
max_iter = 1000

logistic_regression = LogisticRegression(x_train, y_train, normalize_data=False)
theta, cost_histories = logistic_regression.train(max_iter)
labels = logistic_regression.unique_labels

# plt.plot(range(len(cost_histories[0])), cost_histories[0], label=labels[0])
# plt.plot(range(len(cost_histories[1])), cost_histories[1], label=labels[1])
# plt.plot(range(len(cost_histories[2])), cost_histories[2], label=labels[2])
# plt.xlabel('Iteration')
# plt.ylabel('Cost')
# plt.title('Cost Function')
# plt.legend()
# plt.savefig('iris_cost.png')
# plt.show()

#预测
y_train_pred = logistic_regression.predict(x_train)
# print('预测结果: ', y_train_pred)
precision = np.sum(y_train_pred == y_train) / y_train.shape[0] * 100
print('训练集准确率: ', precision, '%')

x_min = np.min(x_train[:, 0])
x_max = np.max(x_train[:, 0])
y_min = np.min(x_train[:, 1])
y_max = np.max(x_train[:, 1])

sample_num = 150
X = np.linspace(x_min, x_max,sample_num)
Y = np.linspace(y_min, y_max,sample_num)

Z_setosa = np.zeros((sample_num, sample_num))
Z_versicolor = np.zeros((sample_num, sample_num))
Z_virginica = np.zeros((sample_num, sample_num))

for x_index, x in enumerate(X):
    for y_index, y in enumerate(Y):
        data = np.array([x, y]).reshape(1, 2)
        prediction = logistic_regression.predict(data)
        Z_setosa[x_index, y_index] = 1 if prediction == labels[0] else 0
        Z_versicolor[x_index, y_index] = 1 if prediction == labels[1] else 0
        Z_virginica[x_index, y_index] = 1 if prediction == labels[2] else 0


for iris_type in iris_types:

    plt.scatter(x_train[(y_train == iris_type).flatten(),0],
                x_train[(y_train == iris_type).flatten(),1],
                label=iris_type)
plt.axis((0.5,7.5,0,3))
plt.contour(X, Y, Z_setosa, levels=1, colors='r')
plt.contour(X, Y, Z_versicolor, levels=1, colors='g')
plt.contour(X, Y, Z_virginica, levels=1, colors='b')
plt.xlabel(x_axis)
plt.ylabel(y_axis)
plt.legend()
plt.savefig('iris_predict.png')
plt.show()



