#Liangyz
#2024/6/4  11:08

import numpy as np

class KMeans:
    def __init__(self,data,num_clustres):
        self.data=data
        self.num_clustres=num_clustres

    def train(self,max_iter):
        centroids= KMeans.centroids_init(self.data,self.num_clustres)#初始化质心,随机选择k个样本作为质心
        num_examples=self.data.shape[0]
        closest_centroids_ids=np.zeros((num_examples,1))
        for _ in range(max_iter):
            #计算每个样本点到质心的距离,并选择最近的质心
            closest_centroids_ids= KMeans.find_closest_centroids(self.data, centroids)
            #重新计算质心
            centroids= KMeans.compute_centroids(self.data,closest_centroids_ids,self.num_clustres)
        return centroids,closest_centroids_ids


    @staticmethod
    def centroids_init(data,num_clustres):
        num_examples=data.shape[0]
        random_ids = np.random.permutation(num_examples)
        centroids = data[random_ids[:num_clustres],:]
        return centroids

    @staticmethod
    def find_closest_centroids(data,centroids):
        num_examples=data.shape[0]
        num_centroids=centroids.shape[0]
        closest_centroids_ids=np.zeros((num_examples,1))
        for example_index in range(num_examples):
            distances=np.zeros((num_centroids,1))
            for centroid_index in range(num_centroids):
                distances[centroid_index]=np.sum((data[example_index,:]-centroids[centroid_index,:])**2)
            closest_centroids_ids[example_index]=np.argmin(distances)
        return closest_centroids_ids

    @staticmethod
    def compute_centroids(data,closest_centroids_ids,num_clustres):
        num_examples=data.shape[0]
        num_features=data.shape[1]
        centroids=np.zeros((num_clustres,num_features))
        for centroid_index in range(num_clustres):
            closest_id = closest_centroids_ids==centroid_index
            centroids[centroid_index]=np.mean(data[closest_id.flatten(),:],axis=0)
        return centroids




