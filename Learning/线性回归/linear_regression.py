#Liangyz
#2024/5/15  16:02

import numpy as np
from .prepare_data import prepare_data

#目标函数: J(theta) = 1/2m * sum((h(x) - y)^2). h(x) = theta^T * x = theta_0 + theta_1 * x_1 + ... + theta_n * x_n
#梯度下降算法: theta_j = theta_j - alpha * 1/m * sum((h(x) - y) * x_j)
#其中: x_0 = 1, x_1, x_2, ..., x_n 为特征值, y 为标签值, m 为样本数, n 为特征数
#data: m * n 的矩阵, m 为样本数, n 为特征数
#labels: m * 1 的矩阵, m 为样本数

class LinearRegression:

    def __init__(self, data, labels, normalize_data=True):
        #准备数据
        (self.data_processed,
         self.feature_mean,
         self.feature_std) = prepare_data(data, normalize_data)
        self.labels = labels
        self.normalized = normalize_data

        num_exmp, num_features = self.data_processed.shape
        self.theta = np.zeros((num_features, 1))

    def train(self, alpha, num_iters=1000):
        """
        训练模型 按照梯度下降算法更新theta
        """
        lost_history = self.gradient_descent(alpha, num_iters)
        return self.theta, lost_history


    def gradient_descent(self, alpha, num_iters):
        """
        梯度下降算法 更新theta
        """
        cost_history = []
        for _ in range(num_iters):
            self.gradient_step(alpha)
            cost_history.append(self.lost_function(self.data_processed, self.labels))
        return cost_history


    def gradient_step(self, alpha):
        """
        梯度下降算法 矩阵计算theta
        """
        num_exmp = self.data_processed.shape[0]
        prediction = self.hypothesis(self.data_processed, self.theta)
        error = prediction - self.labels

        self.theta -= alpha * (1 / num_exmp) * np.dot(self.data_processed.T, error)#h(x) = theta^T * x

    def lost_function(self, data, labels):
        """
        计算损失函数
        """
        num_exmp = data.shape[0]
        prediction = self.hypothesis(data, self.theta)
        error = prediction - labels
        lost = (1 / (2 * num_exmp)) * np.dot(error.T, error)
        return lost

    @staticmethod
    def hypothesis(data, theta):
        """
        得到预测值
        """
        return np.dot(data, theta)

    def get_lost(self, data, labels):
        """
        得到损失值
        """
        data, _, _ = prepare_data(data, self.normalized)
        return self.lost_function(data, labels)

    def predict(self, data):
        """
        按得到的theta 预测
        """
        data, _, _ = prepare_data(data, self.normalized)
        return self.hypothesis(data, self.theta)
