#Liangyz
#2024/5/15  16:59

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go


from linear_regression import LinearRegression

#读取数据
data = pd.read_csv('../../data/day.csv')
train_data = data.sample(frac=0.8)
test_data = data.drop(train_data.index)

#选择特征
input_features_name_1 = 'temp'
input_features_name_2 = 'hum'
output_features_name = 'cnt'

#训练集
train_x = train_data[[input_features_name_1,input_features_name_2]].values
# print(train_x)
train_y = train_data[[output_features_name]].values
# print(train_y)
#Min-Max 标准化y
train_y = (train_y - np.min(train_y)) / (np.max(train_y) - np.min(train_y))



#测试集
test_x = test_data[[input_features_name_1, input_features_name_2]].values
test_y = test_data[[output_features_name]].values
# print(test_y)
#Min-Max 标准化y
test_y = (test_y - np.min(test_y)) / (np.max(test_y) - np.min(test_y))

#数据散点图
# plot_training = go.Scatter3d(x=train_x[:,0].flatten(),
#                              y=train_x[:,1].flatten(),
#                              z=train_y.flatten(),
#                              mode='markers',
#                              marker=dict(size=5,
#                                          color='blue',
#                                          opacity=1),
#                                          name='Training Data')
# plot_test = go.Scatter3d(x=test_x[:,0].flatten(),
#                          y=test_x[:,1].flatten(),
#                          z=test_y.flatten(),
#                          mode='markers',
#                          marker=dict(size=5,
#                                      color='red',
#                                      opacity=1),
#                                      name='Test Data')
# layout = go.Layout(scene=dict(xaxis=dict(title=input_features_name_1),
#                                 yaxis=dict(title=input_features_name_2),
#                                 zaxis=dict(title=output_features_name)),
#                                 margin=dict(l=0,
#                                             r=0,
#                                             b=0,
#                                             t=0))
# fig = go.Figure(data=[plot_training, plot_test], layout=layout)
# plotly.offline.plot(fig, filename='bike_sharing.html')



#训练模型
num_iters = 500
alpha = 0.01
model = LinearRegression(train_x, train_y)
(theta, cost_history) = model.train(alpha, num_iters)
# print(cost_history)

print('开始的损失函数值: ', cost_history[0])
print('结束的损失函数值: ', cost_history[-1])

#损失函数值变化
plot_cost = go.Scatter(y=cost_history,
                        mode='lines',
    name='Cost Function')
layout = go.Layout(xaxis=dict(title='Iteration'),
                    yaxis=dict(title='Cost'))
fig = go.Figure(data=[plot_cost], layout=layout)
plotly.offline.plot(fig, filename='cost_function_2.html')



#预测
prediction_num = 100
x1 = np.linspace(np.min(train_x[:,0]), np.max(train_x[:,0]), prediction_num)
x2 = np.linspace(np.min(train_x[:,1]), np.max(train_x[:,1]), prediction_num)

x_pred = np.zeros((prediction_num * prediction_num, 1))
y_pred = np.zeros((prediction_num * prediction_num, 1))

x_y_index = 0
for x_index, x_value in enumerate(x1):
    for y_index, y_value in enumerate(x2):
        x_pred[x_y_index] = x_value
        y_pred[x_y_index] = y_value
        x_y_index += 1

z_pred = model.predict(np.hstack((x_pred, y_pred)))


#预测结果图
plot_training = go.Scatter3d(x=train_x[:,0].flatten(),
                                y=train_x[:,1].flatten(),
                                z=train_y.flatten(),
                                mode='markers',
                                marker=dict(size=5,
                                            color='blue',
                                            opacity=1),
                                name='Training Data')
plot_test = go.Scatter3d(x=test_x[:,0].flatten(),
                            y=test_x[:,1].flatten(),
                            z=test_y.flatten(),
                            mode='markers',
                            marker=dict(size=5,
                                        color='red',
                                        opacity=1),
                                        name='Test Data')
plot_pred = go.Scatter3d(x=x_pred.flatten(),
                            y=y_pred.flatten(),
                            z=z_pred.flatten(),
                            mode='markers',
                            marker=dict(size=5,
                                        color='green',
                                        opacity=1),
                                        name='Prediction')
layout = go.Layout(scene=dict(xaxis=dict(title=input_features_name_1),
                                yaxis=dict(title=input_features_name_2),
                                zaxis=dict(title=output_features_name)),
                                margin=dict(l=0,
                                            r=0,
                                            b=0,
                                            t=0))
fig = go.Figure(data=[plot_training, plot_test, plot_pred], layout=layout)
plotly.offline.plot(fig, filename='bike_sharing_prediction_2.html')


