#Liangyz
#2024/6/5  23:10

import numpy as np
from Learning.utils.features import prepare_for_training
from Learning.utils.hypothesis import sigmoid, sigmoid_gradient


class MultilayerPerceptron:
    def __init__(self, data, labels, layers, normalize_data=False):
        (data_prepared,
        _,
        _)= prepare_for_training(data, normalize_data=normalize_data)#数据预处理
        self.data = data_prepared
        self.labels = labels
        self.layers = layers #784 25 10
        self.normalize_data = normalize_data
        self.theta = MultilayerPerceptron.initialize_theta(layers)

    def predict(self, data):#预测
        (data_prepared,
         _,
         _)=prepare_for_training(data,normalize_data=self.normalize_data)  #数据预处理
        num_examples = data_prepared.shape[0]

        predictions = MultilayerPerceptron.feedforward_propagation(data_prepared, self.layers, self.theta)#前向传播

        return np.argmax(predictions, axis=1).reshape((num_examples, 1))

    def train(self, max_iteration=1000, alpha=0.1):#训练
        unrolled_thetas = MultilayerPerceptron.theta_unroll(self.theta)#将参数矩阵展开成一维向量
        (optimized_theta, cost_history) = MultilayerPerceptron.gradient_descent(self.data, self.labels, self.layers, unrolled_thetas, max_iteration, alpha)#梯度下降
        self.theta = MultilayerPerceptron.theta_roll(optimized_theta, self.layers)#将一维向量展开成参数矩阵
        return self.theta, cost_history

    @staticmethod
    def initialize_theta(layers):#初始化参数矩阵
        num_layers = len(layers)
        thetas = {}#参数矩阵
        for layers_index in range(num_layers - 1):#遍历每一层
            """
            会执行两次, 得到两个参数矩阵: 25*785, 10*26
            """
            in_count = layers[layers_index]
            out_count = layers[layers_index + 1]
            thetas[layers_index] = np.random.rand(out_count, in_count + 1) * 0.05#随机初始化参数矩阵, 0.05是为了保证参数不会太大
            # +1 for bias(偏置),加在输入层的最后一列

        return thetas

    @staticmethod
    def theta_unroll(thetas):#将参数矩阵展开成一维向量
        """
        将参数矩阵展开成一维向量
        """
        num_layers = len(thetas)
        unrolled_thetas = np.array([])#一维向量
        for theta_layer_index in range(num_layers):#遍历每一层
            unrolled_thetas = np.hstack((unrolled_thetas, thetas[theta_layer_index].flatten()))
        return unrolled_thetas

    @staticmethod
    def gradient_descent(data, labels, layers, unrolled_thetas, max_iteration, alpha):#梯度下降
        optimized_theta = unrolled_thetas#优化后的参数矩阵
        cost_history = []#损失函数历史

        for i in range(max_iteration):#迭代
            # if i == max_iteration//2:#迭代次数的一半
            #     alpha = alpha*0.8
            #     cost=MultilayerPerceptron.cost_function(data,labels,layers,
            #                                             MultilayerPerceptron.theta_roll(optimized_theta,
            #                                                                             layers))  #计算损失函数
            #     cost_history.append(cost)
            #     theta_gradient=MultilayerPerceptron.gradient_step(data,labels,optimized_theta,layers,alpha)  #梯度下降
            #     optimized_theta=theta_gradient  #更新参数矩阵
            # else:
            cost = MultilayerPerceptron.cost_function(data, labels, layers, MultilayerPerceptron.theta_roll(optimized_theta, layers))#计算损失函数
            cost_history.append(cost)
            theta_gradient = MultilayerPerceptron.gradient_step(data, labels, optimized_theta, layers, alpha)#梯度下降
            optimized_theta = theta_gradient#更新参数矩阵

        return optimized_theta, cost_history

    @staticmethod
    def gradient_step(data, labels, optimized_theta, layers, alpha):#梯度下降
        theta = MultilayerPerceptron.theta_roll(optimized_theta, layers)#将一维向量展开成参数矩阵
        thetas_rolled_gradients = MultilayerPerceptron.backpropagation(data, labels, layers, theta)#反向传播
        thetas_unrolled_gradients = MultilayerPerceptron.theta_unroll(thetas_rolled_gradients)#将参数矩阵展开成一维向量
        optimized_theta -= alpha * thetas_unrolled_gradients#梯度下降
        return optimized_theta

    @staticmethod
    def backpropagation(data, labels, layers, thetas):#反向传播
        num_layers = len(layers)#层数
        (num_examples, num_features) = data.shape#样本数, 特征数
        num_labels = layers[-1]#标签数

        deltas = {}
        #初始化delta
        for layers_index in range(num_layers - 1):
            in_count = layers[layers_index]#输入层
            out_count = layers[layers_index + 1]#输出层
            deltas[layers_index] = np.zeros((out_count, in_count + 1)) #25*785, 10*26

        for example_index in range(num_examples):#遍历每一个样本
            layers_inputs = {}#输入
            layers_activations = {}#激活
            layers_activations[0] = data[example_index,:].reshape((num_features,1))#输入层
            # Forward propagation
            for layer_index in range(num_layers - 1):#遍历每一层
                thetas_layer = thetas[layer_index] #25*785, 10*26
                layers_inputs[layer_index + 1] = np.dot(thetas_layer, layers_activations[layer_index])#第一次是25*785*785*1=25*1, 第二次是10*26*26*1=10*1
                layers_activations[layer_index + 1] = np.vstack((np.array([1]), sigmoid(layers_inputs[layer_index + 1])))#加上bias
            out_layer_activation = layers_activations[num_layers - 1][1:,:]#去掉bias

            delta = {}
            #标签处理
            bitwise_labels=np.zeros((num_labels,1))#one hot encoding
            bitwise_labels[labels[example_index][0]]=1#one hot encoding
            #计算输出层的差异
            delta[num_layers - 1] = out_layer_activation - bitwise_labels#10*1
            #反向传播
            for layer_index in range(num_layers - 2, 0, -1):#遍历每一层
                layer_theta = thetas[layer_index]#10*26, 25*785
                next_layer_delta = delta[layer_index + 1]#10*1, 25*1
                layer_input = layers_inputs[layer_index]#10*1, 25*1
                layer_input = np.vstack((np.array([1]), layer_input))#加上bias
                #计算当前层的差异
                delta[layer_index] = np.dot(layer_theta.T, next_layer_delta) * sigmoid_gradient(layer_input)#10*26, 25*785
                delta[layer_index] = delta[layer_index][1:,:]#去掉bias
            for layer_index in range(num_layers - 1):#遍历每一层
                layer_delta = np.dot(delta[layer_index + 1], layers_activations[layer_index].T)#10*26, 25*1
                deltas[layer_index] += layer_delta#第一次是25*785, 第二次是10*26

        delta = {key: value / num_examples for key, value in deltas.items()}#求平均
        return delta

    @staticmethod
    def theta_roll(unrolled_thetas, layers):#将一维向量展开成参数矩阵
        """
        将一维向量展开成参数矩阵
        """
        num_layers = len(layers)
        thetas = {}
        unrolled_shift = 0#偏移量
        for layers_index in range(num_layers - 1):
            in_count = layers[layers_index]#输入层
            out_count = layers[layers_index + 1]#输出层

            thetas_width = in_count + 1#theta的宽度
            thetas_height = out_count#theta的高度
            thetas_volume = thetas_width * thetas_height#参数矩阵的大小
            start_index = unrolled_shift#起始位置
            end_index = unrolled_shift + thetas_volume#结束位置
            layer_theta_unrolled = unrolled_thetas[start_index:end_index]#一维向量
            thetas[layers_index] = layer_theta_unrolled.reshape((thetas_height, thetas_width))#参数矩阵
            unrolled_shift = unrolled_shift + thetas_volume

        return thetas

    @staticmethod
    def cost_function(data, labels, layers, thetas):#计算损失函数
        num_layers = len(layers)
        num_examples = data.shape[0]
        num_labels = layers[-1]

        # Forward propagation
        predictions = MultilayerPerceptron.feedforward_propagation(data, layers, thetas)#前向传播
        bitwise_labels = np.zeros((num_examples, num_labels))#标签矩阵
        #制作标签矩阵one hot encoding
        for example_index in range(num_examples):#遍历每一个样本
            bitwise_labels[example_index][labels[example_index][0]] = 1

        bit_set_cost = np.sum(np.log(predictions[bitwise_labels == 1]))#计算损失函数
        bit_unset_cost = np.sum(np.log(1 - predictions[bitwise_labels == 0]))#计算损失函数
        cost = (-1 / num_examples) * (bit_set_cost + bit_unset_cost)

        return cost

    @staticmethod
    def feedforward_propagation(data, layers, thetas):#前向传播
        num_layers=len(layers)
        num_examples=data.shape[0]
        in_layer_activation = data

        #逐层计算
        for layer_index in range(num_layers - 1):
            thetas_layer = thetas[layer_index]
            out_layer_activation = sigmoid(np.dot(in_layer_activation, thetas_layer.T))
            #正常情况下, out_layer_activation的shape应该是(num_examples, layers[layer_index + 1])
            out_layer_activation = np.hstack((np.ones((num_examples, 1)), out_layer_activation))
            in_layer_activation = out_layer_activation

        return in_layer_activation[:, 1:]#去掉bias










