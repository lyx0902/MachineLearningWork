import numpy as np
from sklearn.preprocessing import StandardScaler

# 定义 sigmoid 函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 梯度下降实现 Logistic 回归
def logistic_regression_gd(X, y, alpha=0.01, epochs=1000):
    m, n = X.shape
    theta = np.zeros(n)  # 初始化权重参数
    for epoch in range(epochs):
        z = X @ theta
        h = sigmoid(z)
        gradient = (1 / m) * X.T @ (h - y)
        theta -= alpha * gradient
    return theta

# 随机梯度下降实现 Logistic 回归
def logistic_regression_sgd(X, y, alpha=0.01, epochs=1000):
    m, n = X.shape
    theta = np.zeros(n)  # 初始化权重参数
    for epoch in range(epochs):
        for i in range(m):  # 遍历每个样本
            xi = X[i, :].reshape(1, -1)
            yi = y[i]
            z = xi @ theta
            h = sigmoid(z)
            gradient = (h - yi) * xi
            theta -= alpha * gradient.flatten()
    return theta

# Softmax 函数
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # 防止溢出
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# 梯度下降实现 Softmax 回归
def softmax_regression_gd(X, y, alpha=0.01, epochs=1000, num_classes=None):
    m, n = X.shape
    if num_classes is None:
        num_classes = len(np.unique(y))
    y_onehot = np.eye(num_classes)[y]  # 转换为独热编码
    theta = np.zeros((n, num_classes))  # 初始化权重参数
    for epoch in range(epochs):
        z = X @ theta
        h = softmax(z)
        gradient = (1 / m) * X.T @ (h - y_onehot)
        theta -= alpha * gradient
    return theta

# 数据预处理
def load_and_preprocess_data():
    # 加载数据
    ex4x = np.loadtxt('dataset/ex4x.dat')
    ex4y = np.loadtxt('dataset/ex4y.dat')

    # 标准化数据
    scaler = StandardScaler()
    ex4x_normalized = scaler.fit_transform(ex4x)

    # 添加偏置项
    X = np.c_[np.ones((ex4x_normalized.shape[0], 1)), ex4x_normalized]
    y = ex4y.astype(int)

    return X, y

# 主程序
if __name__ == "__main__":
    X, y = load_and_preprocess_data()

    # Logistic 回归
    theta_logistic_gd = logistic_regression_gd(X, y, alpha=0.1, epochs=1000)
    theta_logistic_sgd = logistic_regression_sgd(X, y, alpha=0.1, epochs=100)

    # Softmax 回归
    theta_softmax = softmax_regression_gd(X, y, alpha=0.1, epochs=1000)

    # 打印结果
    print("Logistic 回归 (梯度下降) 参数:", theta_logistic_gd)
    print("Logistic 回归 (随机梯度下降) 参数:", theta_logistic_sgd)
    print("Softmax 回归参数:", theta_softmax)
