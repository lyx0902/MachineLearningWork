import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
import time

# 初始化权重
def initialize_weights(input_size, hidden_size, output_size):
    w1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros((1, hidden_size))
    w2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros((1, output_size))
    return w1, b1, w2, b2

# 激活函数和导数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# 前向传播
def forward_propagation(X, w1, b1, w2, b2):
    z1 = X @ w1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ w2 + b2
    a2 = softmax(z2)
    return z1, a1, z2, a2

# 反向传播
def backward_propagation(X, y, z1, a1, a2, w2):
    m = X.shape[0]
    y_onehot = np.eye(a2.shape[1])[y]
    dz2 = a2 - y_onehot
    dw2 = (a1.T @ dz2) / m
    db2 = np.sum(dz2, axis=0, keepdims=True) / m

    dz1 = (dz2 @ w2.T) * sigmoid_derivative(a1)
    dw1 = (X.T @ dz1) / m
    db1 = np.sum(dz1, axis=0, keepdims=True) / m

    return dw1, db1, dw2, db2

# 训练 BP 神经网络
def train_neural_network(X, y, hidden_size=10, alpha=0.01, epochs=1000):
    input_size = X.shape[1]
    output_size = len(np.unique(y))

    # 初始化权重
    w1, b1, w2, b2 = initialize_weights(input_size, hidden_size, output_size)

    for epoch in range(epochs):
        # 前向传播
        z1, a1, z2, a2 = forward_propagation(X, w1, b1, w2, b2)

        # 反向传播
        dw1, db1, dw2, db2 = backward_propagation(X, y, z1, a1, a2, w2)

        # 更新权重
        w1 -= alpha * dw1
        b1 -= alpha * db1
        w2 -= alpha * dw2
        b2 -= alpha * db2

    return w1, b1, w2, b2

# 评估模型
def evaluate_neural_network(X, y, w1, b1, w2, b2):
    _, _, _, a2 = forward_propagation(X, w1, b1, w2, b2)
    y_pred = np.argmax(a2, axis=1)
    accuracy = accuracy_score(y, y_pred)
    loss = log_loss(y, a2)
    return accuracy, loss

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

    # 训练 BP 神经网络
    start_time = time.time()
    w1, b1, w2, b2 = train_neural_network(X, y, hidden_size=10, alpha=0.1, epochs=1000)
    time_nn = time.time() - start_time

    # 评估 BP 神经网络
    acc_nn, loss_nn = evaluate_neural_network(X, y, w1, b1, w2, b2)

    # 打印结果
    print("BP 神经网络:")
    print(f"准确率: {acc_nn:.4f}, 损失值: {loss_nn:.4f}, 收敛时间: {time_nn:.4f}s")
