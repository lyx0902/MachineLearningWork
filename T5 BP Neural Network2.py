import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# Load data from ex4x.dat and ex4y.dat
X = np.loadtxt('dataset/ex4x.dat')
y = np.loadtxt('dataset/ex4y.dat')

# Normalize the data
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

hidden_layer_sizes = [10, 20, 30]  # 不同的隐藏层大小
activation_functions = ['sigmoid', 'relu']  # 不同的激活函数
learning_rates = [0.001,0.01,0.1,0.5]  # 不同的学习率
epochs = 100

results = []

for hidden_layer_size in hidden_layer_sizes:
    for activation_function in activation_functions:
        for learning_rate in learning_rates:
            accuracy_scores = []

            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                # Build the model
                model = Sequential([
                    Input(shape=(X.shape[1],)),
                    Dense(hidden_layer_size, activation=activation_function),
                    Dense(1, activation='sigmoid')
                ])

                # Compile the model
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                              loss='binary_crossentropy',
                              metrics=['accuracy'])

                # Train the model
                model.fit(X_train, y_train, epochs=epochs, verbose=0, batch_size=32)

                # Evaluate the model
                loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
                accuracy_scores.append(accuracy)

            mean_accuracy = np.mean(accuracy_scores)
            results.append((hidden_layer_size, activation_function, learning_rate, mean_accuracy))
            print(f'Hidden Layer: {hidden_layer_size}, Activation: {activation_function}, Learning Rate: {learning_rate}, Accuracy: {mean_accuracy:.2f}')

# Print best configuration
best_config = max(results, key=lambda x: x[3])
print(f'Best Configuration: Hidden Layer: {best_config[0]}, Activation: {best_config[1]}, Learning Rate: {best_config[2]}, Accuracy: {best_config[3]:.2f}')