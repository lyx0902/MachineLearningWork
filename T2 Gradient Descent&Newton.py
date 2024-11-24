import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 定义 sigmoid 函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 损失函数 (交叉熵)
def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    cost = -(1 / m) * (y @ np.log(h) + (1 - y) @ np.log(1 - h))
    return cost

# 梯度下降实现 Logistic 回归
def logistic_regression_gd(X, y, alpha=0.01, epochs=50):
    m, n = X.shape
    theta = np.zeros(n)
    cost_history = []
    for epoch in range(epochs):
        z = X @ theta
        h = sigmoid(z)
        gradient = (1 / m) * X.T @ (h - y)
        theta -= alpha * gradient
        cost_history.append(compute_cost(X, y, theta))
    return theta, cost_history

# 随机梯度下降实现 Logistic 回归
def logistic_regression_sgd(X, y, alpha=0.01, epochs=50):
    m, n = X.shape
    theta = np.zeros(n)
    cost_history = []
    for epoch in range(epochs):
        for i in range(m):
            xi = X[i, :].reshape(1, -1)
            yi = y[i]
            z = xi @ theta
            h = sigmoid(z)
            gradient = (h - yi) * xi
            theta -= alpha * gradient.flatten()
        cost_history.append(compute_cost(X, y, theta))
    return theta, cost_history

# 牛顿法实现 Logistic 回归
def logistic_regression_newton(X, y, epochs=50):
    m, n = X.shape
    theta = np.zeros(n)
    cost_history = []
    for epoch in range(epochs):
        z = X @ theta
        h = sigmoid(z)
        gradient = (1 / m) * X.T @ (h - y)
        H = (1 / m) * X.T @ np.diag(h * (1 - h)) @ X  # Hessian 矩阵
        theta -= np.linalg.inv(H) @ gradient
        cost_history.append(compute_cost(X, y, theta))
    return theta, cost_history

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

# 绘制损失变化曲线
def plot_cost_history(cost_history, method):
    plt.plot(cost_history, label=method)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title(f'Cost vs. Iterations ({method})')
    plt.legend()

# 绘制决策边界
def plot_decision_boundary(X, y, theta, method):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y == 0, 1], X[y == 0, 2], label="Not Admitted", marker='o', color='blue')
    plt.scatter(X[y == 1, 1], X[y == 1, 2], label="Admitted", marker='+', color='red')

    # 决策边界: 𝜃₀ + 𝜃₁x₁ + 𝜃₂x₂ = 0 -> x₂ = -(𝜃₀ + 𝜃₁x₁) / 𝜃₂
    x_values = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    y_values = -(theta[0] + theta[1] * x_values) / theta[2]
    plt.plot(x_values, y_values, label='Decision Boundary', color='green')

    plt.xlabel('Exam 1 Score (Standardized)')
    plt.ylabel('Exam 2 Score (Standardized)')
    plt.title(f'Decision Boundary ({method})')
    plt.legend()

# 主程序
if __name__ == "__main__":
    X, y = load_and_preprocess_data()

    # 梯度下降
    theta_gd, cost_history_gd = logistic_regression_gd(X, y, alpha=0.1, epochs=50)

    # 随机梯度下降
    theta_sgd, cost_history_sgd = logistic_regression_sgd(X, y, alpha=0.1, epochs=50)

    # 牛顿法
    theta_newton, cost_history_newton = logistic_regression_newton(X, y, epochs=50)

    # 打印最终结果
    print("Logistic 回归 (梯度下降) 参数:", theta_gd)
    print("Logistic 回归 (随机梯度下降) 参数:", theta_sgd)
    print("Logistic 回归 (牛顿法) 参数:", theta_newton)

    # 绘制损失函数曲线
    plt.figure(figsize=(8, 6))
    plot_cost_history(cost_history_gd, "Gradient Descent")
    plt.savefig("T2Graphs/cost_gd.png")
    plt.show()

    plt.figure(figsize=(8, 6))
    plot_cost_history(cost_history_sgd, "Stochastic Gradient Descent")
    plt.savefig("T2Graphs/cost_sgd.png")
    plt.show()

    plt.figure(figsize=(8, 6))
    plot_cost_history(cost_history_newton, "Newton's Method")
    plt.savefig("T2Graphs/cost_newton.png")
    plt.show()

    # 绘制决策边界
    plot_decision_boundary(X, y, theta_gd, "Gradient Descent")
    plt.savefig("T2Graphs/boundary_gd.png")
    plt.show()

    plot_decision_boundary(X, y, theta_sgd, "Stochastic Gradient Descent")
    plt.savefig("T2Graphs/boundary_sgd.png")
    plt.show()

    plot_decision_boundary(X, y, theta_newton, "Newton's Method")
    plt.savefig("T2Graphs/boundary_newton.png")
    plt.show()
