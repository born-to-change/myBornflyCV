from sklearn.preprocessing import StandardScaler
data = [[0, 0], [0, 0], [1, 1], [1, 1]]
scaler = StandardScaler()
print(scaler.fit(data)) #计算data的均值和标准差，存到saler中
StandardScaler(copy=True, with_mean=True, with_std=True)
print(scaler.mean_)
print(scaler.var_)
print(scaler)
print(scaler.transform(data))
print(scaler.transform([[2, 2]])) #用scaler得到的均值和标准差，标准化【2，2】
# StandardScaler(copy=True, with_mean=True, with_std=True)
# [0.5 0.5]
# [0.25 0.25]
# StandardScaler(copy=True, with_mean=True, with_std=True)
# [[-1. - 1.]
#  [-1. - 1.]
#  [1.  1.]
# [1.
# 1.]]
# [[3. 3.]]