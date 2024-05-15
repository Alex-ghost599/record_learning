#Liangyz
#2024/5/14  21:17

import numpy as np
#-----------------------Z-Score 标准化-------------
# 示例数据：m 个 n 维向量
vectors = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12]
])

# 计算向量的均值
mean_vector = np.mean(vectors, axis=0)
std_vector = np.std(vectors, axis=0)

# Z-Score 标准化
standardized_vectors = (vectors - mean_vector) / std_vector

print("Z-Score 标准化:\n", standardized_vectors)

# -----------------------Min-Max 标准化-----------

# 示例数据：m 个 n 维向量
vectors = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12]
])

# 计算最小值和最大值
min_vector = np.min(vectors, axis=0)
max_vector = np.max(vectors, axis=0)

# Min-Max 标准化
min_max_vectors = (vectors - min_vector) / (max_vector - min_vector)

print("Min-Max 标准化:\n", min_max_vectors)

# ------------------------------最大绝对值标准化------------------

# 示例数据：m 个 n 维向量
vectors = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12]
])

# 计算最大绝对值
max_abs_vector = np.max(np.abs(vectors), axis=0)

# 最大绝对值标准化
max_abs_vectors = vectors / max_abs_vector

print("最大绝对值标准化:\n", max_abs_vectors)


