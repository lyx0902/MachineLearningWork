import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 指定支持中文的字体
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
# sigmoid激活函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# softmax激活函数
def softmax(z):
    # 防止溢出
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# 计算Logistic回归的损失
def compute_cost_logistic(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    return -1/m * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

# Logistic回归的梯度下降优化
def logistic_gd(X, y, alpha=0.1, tol=1e-3):
    m, n = X.shape
    theta = np.zeros(n)
    cost_history = []
    while True:
        h = sigmoid(X @ theta)
        gradient = (1/m) * X.T @ (h - y)
        theta -= alpha * gradient
        cost = compute_cost_logistic(X, y, theta)
        cost_history.append(cost)
        # 如果连续两次迭代的损失函数值变化小于tol，则停止迭代
        if len(cost_history) > 1 and abs(cost_history[-1] - cost_history[-2]) < tol:
            break
    return theta, cost_history

# Logistic回归的随机梯度下降优化
def logistic_sgd(X, y, alpha=0.1, tol=1e-3, batch_size=5):
    m, n = X.shape
    theta = np.zeros(n)
    cost_history = []
    while True:
        indices = np.random.choice(m, batch_size, replace=False)
        X_batch = X[indices]
        y_batch = y[indices]
        h = sigmoid(X_batch @ theta)
        gradient = (1/batch_size) * X_batch.T @ (h - y_batch)
        theta -= alpha * gradient
        cost = compute_cost_logistic(X, y, theta)
        cost_history.append(cost)
        if len(cost_history) > 1 and abs(cost_history[-1] - cost_history[-2]) < tol:
            break
    return theta, cost_history

# 计算Softmax回归的损失
def compute_cost_softmax(X, y, theta, num_classes):
    m = len(y)
    logits = X @ theta
    probs = softmax(logits)
    log_probs = -np.log(probs[np.arange(m), y])
    return np.sum(log_probs) / m

# Softmax回归的梯度下降优化
def softmax_gd(X, y, num_classes, alpha=0.1, tol=1e-3):
    m, n = X.shape
    theta = np.zeros((n, num_classes))
    cost_history = []
    while True:
        logits = X @ theta
        probs = softmax(logits)
        gradient = (1/m) * X.T @ (probs - np.eye(num_classes)[y])
        theta -= alpha * gradient
        cost = compute_cost_softmax(X, y, theta, num_classes)
        cost_history.append(cost)
        if len(cost_history) > 1 and abs(cost_history[-1] - cost_history[-2]) < tol:
            break
    return theta, cost_history

# Softmax回归的随机梯度下降优化
def softmax_sgd(X, y, num_classes, alpha=0.1, tol=1e-3):
    m, n = X.shape
    theta = np.zeros((n, num_classes))
    cost_history = []
    while True:
        i = np.random.randint(m)
        X_i = X[i:i+1]
        y_i = y[i]
        logits = X_i @ theta
        probs = softmax(logits)
        gradient = X_i.T @ (probs - np.eye(num_classes)[y_i])
        theta -= alpha * gradient
        cost = compute_cost_softmax(X, y, theta, num_classes)
        cost_history.append(cost)
        if len(cost_history) > 1 and abs(cost_history[-1] - cost_history[-2]) < tol:
            break
    return theta, cost_history

# 绘制损失函数历史图
def plot_cost_history(cost_history, title):
    plt.figure()
    plt.plot(range(1, len(cost_history)+1), cost_history, marker='o')
    plt.xlabel('迭代次数', fontproperties=font)
    plt.ylabel('损失值', fontproperties=font)
    plt.title(title, fontproperties=font)
    plt.show()

# 绘制Logistic回归的决策边界
def plot_decision_boundary(X, y, theta, title):
    plt.figure()
    plt.scatter(X[y == 0][:, 1], X[y == 0][:, 2], marker='o', label='类别 0', color='blue')
    plt.scatter(X[y == 1][:, 1], X[y == 1][:, 2], marker='+', label='类别 1', color='red')
    x_min, x_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    y_min, y_max = X[:, 2].min() - 1, X[:, 2].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    z = sigmoid(np.c_[np.ones(xx.ravel().shape), xx.ravel(), yy.ravel()] @ theta)
    z = z.reshape(xx.shape)
    plt.contour(xx, yy, z, levels=[0.5], colors='black')
    plt.xlabel('特征 1', fontproperties=font)
    plt.ylabel('特征 2', fontproperties=font)
    plt.legend(prop=font)
    plt.title(title, fontproperties=font)
    plt.show()

# 绘制Softmax回归的决策边界
def plot_softmax_decision_boundary(X, y, theta, num_classes, title):
    plt.figure()
    colors = ['blue', 'red', 'green']
    markers = ['o', '+', 'x']
    for i in range(num_classes):
        plt.scatter(X[y == i, 1], X[y == i, 2], marker=markers[i], label=f"类别 {i}", color=colors[i])
    x_min, x_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    y_min, y_max = X[:, 2].min() - 1, X[:, 2].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    grid = np.c_[np.ones(xx.ravel().shape), xx.ravel(), yy.ravel()]
    z = np.dot(grid, theta)
    predictions = np.argmax(softmax(z), axis=1)
    predictions = predictions.reshape(xx.shape)
    plt.contourf(xx, yy, predictions, alpha=0.2, colors=colors[:num_classes])
    plt.xlabel('特征 1', fontproperties=font)
    plt.ylabel('特征 2', fontproperties=font)
    plt.legend(prop=font)
    plt.title(title, fontproperties=font)
    plt.show()

# 加载数据
X = np.loadtxt("dataset/ex4x.dat")
y = np.loadtxt("dataset/ex4y.dat").astype(int)
# 添加偏置项并归一化特征
X = (X - X.mean(axis=0)) / X.std(axis=0)
X = np.c_[np.ones(X.shape[0]), X]

# Logistic回归（梯度下降）
theta_logistic_gd, cost_history_logistic_gd = logistic_gd(X, y)
plot_cost_history(cost_history_logistic_gd, "Logistic GD - 损失函数收敛")
plot_decision_boundary(X, y, theta_logistic_gd, "Logistic GD - 决策边界")

# Logistic回归（随机梯度下降）
theta_logistic_sgd, cost_history_logistic_sgd = logistic_sgd(X, y)
plot_cost_history(cost_history_logistic_sgd, "Logistic SGD - 损失函数收敛")
plot_decision_boundary(X, y, theta_logistic_sgd, "Logistic SGD - 决策边界")

#  准备Softmax回归
y_softmax = y
num_classes = len(np.unique(y_softmax))

#  Softmax回归（梯度下降）
theta_softmax_gd, cost_history_softmax_gd = softmax_gd(X, y_softmax, num_classes)
plot_cost_history(cost_history_softmax_gd, "Softmax GD - 损失函数收敛")
plot_softmax_decision_boundary(X, y_softmax, theta_softmax_gd, num_classes, "Softmax GD - 决策边界")

#  Softmax回归（随机梯度下降）
theta_softmax_sgd, cost_history_softmax_sgd = softmax_sgd(X, y_softmax, num_classes)
plot_cost_history(cost_history_softmax_sgd, "Softmax SGD - 损失函数收敛")
plot_softmax_decision_boundary(X, y_softmax, theta_softmax_sgd, num_classes, "Softmax SGD - 决策边界")
