#Liangyz
#2024/5/15  16:59

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from linear_regression import LinearRegression

#读取数据
data = pd.read_csv('../../data/day.csv')
train_data = data.sample(frac=0.8)
test_data = data.drop(train_data.index)

#选择特征
input_features_name = 'temp'
output_features_name = 'cnt'

#训练集
train_x = train_data[[input_features_name]].values
# print(train_x)
train_y = train_data[[output_features_name]].values
# print(train_y)
#Min-Max 标准化y
train_y = (train_y - np.min(train_y)) / (np.max(train_y) - np.min(train_y))



#测试集
test_x = test_data[[input_features_name]].values
test_y = test_data[[output_features_name]].values
# print(test_y)
#Min-Max 标准化y
test_y = (test_y - np.min(test_y)) / (np.max(test_y) - np.min(test_y))

#数据散点图
# plt.scatter(train_x, train_y, color='blue', label='train')
# plt.scatter(test_x, test_y, color='red', label='test')
# plt.xlabel(input_features_name)
# plt.ylabel(output_features_name)
# plt.title('Bike Sharing')
# plt.legend()
# plt.show()

#训练模型
num_iters = 500
alpha = 0.01
model = LinearRegression(train_x, train_y)
(theta, cost_history) = model.train(alpha, num_iters)
# print(cost_history)

print('开始的损失函数值: ', cost_history[0])
print('结束的损失函数值: ', cost_history[-1])

#损失函数值变化图
plt.plot(range(num_iters), cost_history)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost Function')
plt.show()

#预测
prediction_num = 100
x_predict = np.linspace(test_x.min(), test_x.max(), prediction_num).reshape(-1, 1)
y_predict = model.predict(x_predict)

#预测结果图
plt.scatter(train_x, train_y, color='b', label='train')
plt.scatter(test_x, test_y, color='g', label='test')
plt.plot(x_predict, y_predict, color='r', label='predict')
plt.xlabel(input_features_name)
plt.ylabel(output_features_name)
plt.title('Bike Sharing')
plt.legend()
plt.show()
plt.savefig('Bike_Sharing_one_features.png')


