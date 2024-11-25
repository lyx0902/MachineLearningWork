import numpy as np
from sklearn.preprocessing import StandardScaler


def loadDataSet():
    dataMat = []
    labelMat = []
    with open('dataset/ex4x.dat') as fr:
        for line in fr.readlines():
            lineArr = line.strip().split()
            dataMat.append([1, float(lineArr[0]), float(lineArr[1])])
    with open('dataset/ex4y.dat') as fr:
        for line in fr.readlines():
            labelMat.append(float(line.strip()))
    return dataMat, labelMat

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression_gd(X, y, alpha=0.01, epochs=1000):
    m, n = X.shape
    theta = np.matrix(np.random.randn(n,1))  # 初始化权重参数
    cost_list = []
    for epoch in range(epochs):
        z = X @ theta
        h = sigmoid(z)
        gradient = (1 / m) * X.T @ (h - y)
        cur_cost = -1 / m * np.sum(np.multiply(y, np.log(h)) + np.multiply((1 - y), np.log(1 - h)))
        if epoch % 50 == 0:
            cost_list.append(cur_cost)
        theta -= alpha * gradient
    return theta, np.array(cost_list)

# 加载数据
X, y = loadDataSet()
X = np.array(X)
y = np.array(y).reshape(-1, 1)

# 数据标准化（可选）
scaler = StandardScaler()
X[:, 1:] = scaler.fit_transform(X[:, 1:])

# 执行梯度下降
theta, cost_list = logistic_regression_gd(X, y)
print("优化后的参数:", theta)
print("损失值列表:", cost_list)
