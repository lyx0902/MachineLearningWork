import numpy as np

# 数据
x = np.array([5.50, 6.75, 7.25, 7.00, 6.50, 5.25, 6.00, 4.50, 8.25, 9.50]).reshape(-1, 1) # 单位：百万
y = np.array([11.50, 13.70, 14.83, 14.15, 13.06, 11.71, 12.16, 9.96, 15.88, 18.33])

# 计算均方误差 (MSE) 函数
def calculate_mse(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    return mse

# 方法一：闭式解（正规方程）
X_b = np.c_[np.ones((len(x), 1)), x]  # 增加一列1表示截距
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y) # theta = (X^T * X)^(-1) * X^T * y
print(f"闭式解的参数：theta0 = {theta_best[0]:.4f}, theta1 = {theta_best[1]:.4f}") # theta0 和 theta1

# 预测广告投入为 1000 万元时的销量
x_new = np.array([[1, 10]])  # 增加1表示截距，广告投入为1000万元，即 x = 10
y_pred_closed_form = x_new.dot(theta_best) # y = theta0 + theta1 * x
print(f"闭式解预测销量（广告投入1000万元）：{y_pred_closed_form[0]:.4f} 单位：百万") # 预测销量

# 计算闭式解的 MSE
y_pred_cf_all = X_b.dot(theta_best)  # 用闭式解计算所有样本的预测值
mse_closed_form = calculate_mse(y, y_pred_cf_all)
print(f"闭式解的均方误差 (MSE)：{mse_closed_form:.4f}")

# 方法二：梯度下降法
def compute_loss(theta, X, y):
    m = len(y)
    return (1 / (2 * m)) * np.sum((X.dot(theta) - y) ** 2)

def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    loss_history = []
    for _ in range(iterations):
        gradients = (1 / m) * X.T.dot(X.dot(theta) - y)
        theta -= learning_rate * gradients
        loss_history.append(compute_loss(theta, X, y))
    return theta, loss_history

# 初始化参数
theta = np.random.randn(2, 1)  # theta0 和 theta1
X_b = np.c_[np.ones((len(x), 1)), x]  # 增加一列1表示截距
y = y.reshape(-1, 1)
learning_rate = 0.01
iterations = 10000

theta_gd, loss_history = gradient_descent(X_b, y, theta, learning_rate, iterations)
print(f"梯度下降法的参数：theta0 = {theta_gd[0, 0]:.4f}, theta1 = {theta_gd[1, 0]:.4f}")

# 预测广告投入为 1000 万元时的销量
y_pred_gd = np.array([[1, 10]]).dot(theta_gd)
print(f"梯度下降法预测销量（广告投入1000万元）：{y_pred_gd[0, 0]:.4f} 单位：百万")

# 计算梯度下降法的 MSE
y_pred_gd_all = X_b.dot(theta_gd)  # 用梯度下降法计算所有样本的预测值
mse_gradient_descent = calculate_mse(y, y_pred_gd_all)
print(f"梯度下降法的均方误差 (MSE)：{mse_gradient_descent:.4f}")
