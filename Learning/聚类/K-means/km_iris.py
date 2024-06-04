#Liangyz
#2024/6/4  15:01

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from K_means import KMeans
from sklearn.datasets import load_iris

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

iris_types = data['class'].unique()

x_axis = iris.feature_names[2]
y_axis = iris.feature_names[3]

plt.figure(figsize=(16, 8))

plt.subplot(121)
for iris_type in iris_types:
    plt.scatter(data[x_axis][data['class'] == iris_type],
                data[data['class'] == iris_type][y_axis],
                label=iris_type)
plt.title('Original Data with Labels')
plt.xlabel(x_axis)
plt.ylabel(y_axis)
plt.legend()

plt.subplot(122)
plt.scatter(data[x_axis], data[y_axis])
plt.title('Original Data without Labels')

plt.xlabel(x_axis)
plt.ylabel(y_axis)
plt.savefig('iris_Original_Data.png')
plt.show()

# ---------------------------------------------train---------------------------------------------------
num_examples = data.shape[0]
x_train = data[[x_axis, y_axis]].values.reshape((num_examples, 2))#(150,2) 2是特征数


num_clusters = 3
max_iter = 50

k_means = KMeans(x_train, num_clusters)
centroids, closest_centroids_ids = k_means.train(max_iter)


plt.figure(figsize=(16, 8))

plt.subplot(121)
for iris_type in iris_types:
    plt.scatter(data[x_axis][data['class'] == iris_type],
                data[data['class'] == iris_type][y_axis],
                label=iris_type)
plt.title('Original Data with Labels')
plt.xlabel(x_axis)
plt.ylabel(y_axis)
plt.legend()

plt.subplot(122)
for centroid_index in range(num_clusters):
    plt.scatter(x_train[(closest_centroids_ids == centroid_index).flatten()][:, 0],
                x_train[(closest_centroids_ids == centroid_index).flatten()][:, 1],
                label='Centroid ' + str(centroid_index))
    plt.scatter(centroids[centroid_index][0], centroids[centroid_index][1], s=200, c='red', marker='x')


"""
for centroid_id, centroid in enumerate(centroids):
    current_examples_ids = (closest_centroids_ids == centroid_id).flatten()
    plt.scatter(x_train[current_examples_ids][:, 0], 
                x_train[current_examples_ids][:, 1], 
                label='Centroid ' + str(centroid_id))

"""

plt.title('K-means Clustering')
plt.xlabel(x_axis)
plt.ylabel(y_axis)
plt.legend()

plt.show()










