#Liangyz
#2024/5/15  15:09
#prepares the data for trainning

import numpy as np
from Learning.utils.normalize import normalize


def prepare_data(data,normalize_data=True):

    #计算样本数
    num_exmp = data.shape[0]
    data_processed = np.copy(data)

    #预处理
    feature_mean = 0
    feature_std = 0
    data_normalized = data_processed
    if normalize_data:
        (data_normalized, feature_mean, feature_std) = normalize(data_processed)

        data_processed = data_normalized

    #添加偏置项(x_0 = 1)
    data_processed = np.hstack((np.ones((num_exmp, 1)), data_processed))

    return data_processed, feature_mean, feature_std
