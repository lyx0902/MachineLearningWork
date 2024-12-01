import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

# 1. 加载数据
ex4x = np.loadtxt("dataset/ex4x.dat")
ex4y = np.loadtxt("dataset/ex4y.dat")

# 2. 数据预处理：标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(ex4x)


# 3. 感知机模型
class Perceptron:
    def __init__(self, learning_rate=0.1, max_iter=50):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = None
        self.bias = 0
        self.losses = []

    def fit(self, X, y):
        # 初始化权重
        self.weights = np.zeros(X.shape[1])
        # 训练过程
        for epoch in range(self.max_iter):
            total_loss = 0
            for i in range(len(X)):
                prediction = np.dot(X[i], self.weights) + self.bias
                # 感知机的判定规则
                if y[i] * prediction <= 0:
                    self.weights += self.learning_rate * y[i] * X[i]
                    self.bias += self.learning_rate * y[i]
                    total_loss += 1  # 每次误分类会累加损失
            self.losses.append(total_loss)

    def predict(self, X):
        return np.sign(np.dot(X, self.weights) + self.bias)


# 4. 创建并训练感知机模型
perceptron = Perceptron(learning_rate=0.1, max_iter=50)
perceptron.fit(X_scaled, ex4y)


# 5. 绘制感知机的决策边界及二分类图像
def plot_decision_boundary(X, y, model):
    plt.figure(figsize=(10, 6))
    h = .02  # 步长
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 绘制决策边界
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=50)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Logistic - Decision Boundary')
    plt.show()


plot_decision_boundary(X_scaled, ex4y, perceptron)

# 6. 绘制损失函数曲线
plt.plot(perceptron.losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Perceptron - Loss Function')
plt.show()

# 7. 实现基于SGD的Logistic回归
log_reg = LogisticRegression(solver='saga', max_iter=50)
log_reg.fit(X_scaled, ex4y)

# 8. 绘制Logistic回归的决策边界
plot_decision_boundary(X_scaled, ex4y, log_reg)

# 9. 对比分析
print("感知机模型训练准确率: {:.2f}".format(np.mean(perceptron.predict(X_scaled) == ex4y)))
print("Logistic回归训练准确率: {:.2f}".format(log_reg.score(X_scaled, ex4y)))

