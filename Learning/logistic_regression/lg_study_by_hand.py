#Liangyz
#2024/6/3  15:06

import numpy as np
from scipy.optimize import minimize
from Learning.utils.prepare_data import prepare_data
from Learning.utils.features.prepare_for_training import prepare_for_training
from Learning.utils.sigmoid import sigmoid


class LogisticRegression:
    def __init__(self,data,labels,polynomial_degree=0,sinusoid_degree=0,normalize_data=True):
        if polynomial_degree==0 and sinusoid_degree==0:
            (self.data,
             self.further_mean,
             self.further_std)=prepare_data(data,normalize_data)
        else:
            (self.data,
             self.further_mean,
             self.further_std)=prepare_for_training(data,polynomial_degree,sinusoid_degree,normalize_data)

        self.labels=labels
        self.unique_labels=np.unique(labels)
        self.normalized=normalize_data
        self.polynomial_degree=polynomial_degree
        self.sinusoid_degree=sinusoid_degree

        num_features=self.data.shape[1]#特征数
        num_unique_labels=np.unique(labels).shape[0]#标签数
        self.theta=np.zeros((num_unique_labels,num_features))#theta

    def train(self, max_inter=1000):#训练模型
        cost_historirs = []
        num_features=self.data.shape[1]
        for label_index, unique_label in enumerate(self.unique_labels):
            current_initial_theta = np.copy(self.theta[label_index])
            current_labels = np.where(self.labels == unique_label, 1, 0)
            (current_theta, cost_history) = LogisticRegression.gradient_descent(self.data, current_labels,current_initial_theta, max_inter)
            self.theta[label_index] = current_theta.T
            cost_historirs.append(cost_history)

        return self.theta, cost_historirs

    @staticmethod
    def gradient_descent(data, labels, initial_theta, max_inter):#梯度下降算法
        cost_history = []
        num_features = data.shape[1]
        minimize_result = minimize(#优化目标:
                                    lambda current_theta:LogisticRegression.cost_function(data,labels,current_theta.reshape(num_features,1)),
                                    #initial_theta:初始值
                                    initial_theta,
                                    #method:优化方法
                                    method='CG',
                                    #jac:是否返回梯度
                                    jac=lambda current_theta:LogisticRegression.gradient_step(data,labels,current_theta.reshape(num_features,1)),
                                    #callback:回调函数
                                    callback=lambda current_theta:cost_history.append(LogisticRegression.cost_function(data,labels,current_theta.reshape(num_features,1))),
                                    #options:参数
                                    options={'maxiter':max_inter})
        if not minimize_result.success:
            raise ArithmeticError('Can not minimize cost function'+minimize_result.message)

        optimized_theta=minimize_result.x.reshape(num_features,1)
        return optimized_theta,cost_history

    @staticmethod
    def cost_function(data,labels,theta):#计算损失函数
        num_exmp=data.shape[0]#样本数
        prediction=LogisticRegression.hypothesis(data,theta)#预测
        y_is_1_cost=np.dot(labels[labels==1].T, np.log(prediction[labels==1]))#y=1时的损失
        y_is_0_cost=np.dot(1-labels[labels==0].T, np.log(1-prediction[labels==0]))#y=0时的损失
        cost=-1/num_exmp*(y_is_1_cost+y_is_0_cost)#损失函数
        return cost

    @staticmethod
    def hypothesis(data,theta):#假设函数
        predictions=sigmoid(np.dot(data,theta))#带入sigmoid函数求预测概率

        return predictions

    @staticmethod
    def gradient_step(data,labels,theta):#梯度下降算法 矩阵计算theta
        num_exmp=labels.shape[0]#样本数
        predictions=LogisticRegression.hypothesis(data,theta)#预测
        error=predictions-labels#误差
        gradient=1/num_exmp*np.dot(data.T,error)#梯度

        return gradient.T.flatten()

    def predict(self,data):#预测
        num_exmp = data.shape[0]#样本数
        if self.polynomial_degree==0 and self.sinusoid_degree==0:
            data_processed = prepare_data(data,self.normalized)[0]#预处理
        else:
            data_processed = prepare_for_training(data,self.polynomial_degree,self.sinusoid_degree,self.normalized)[0]
        prob = LogisticRegression.hypothesis(data_processed,self.theta.T)#预测概率
        max_prob_index = np.argmax(prob,axis=1)#找出最大概率的索引argmax:返回最大值的标签
        class_prediction = np.empty(max_prob_index.shape, dtype=object)#创建一个空数组,准备填写预测索引对应的标签
        for index,label in enumerate(self.unique_labels):#找出实际名称对应的索引
            class_prediction[max_prob_index==index]=label#对应索引填写标签

        return class_prediction.reshape((num_exmp,1))

