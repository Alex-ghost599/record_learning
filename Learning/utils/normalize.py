#Liangyz
#2024/5/15  15:24

#normalize date
import numpy as np


def normalize(features):
    features_normalized=np.copy(features).astype(float)
    #均值
    features_mean=np.mean(features,axis=0)
    #标准差
    features_std=np.std(features,axis=0)

    #标准化
    if features.shape[0]>1:
        features_normalized-=features_mean
        features_std[features_std==0]=1
        features_normalized/=features_std

    return features_normalized,features_mean,features_std
