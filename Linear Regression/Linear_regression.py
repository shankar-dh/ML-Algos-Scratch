import numpy as np

class LinearRegression:

    def __init__(self, X, y, iterations=10000, alpha=0.01):
        self.iterations = iterations
        self.alpha = alpha
        self.weights = np.zeros((X.shape[1] + 1, 1))  # Include bias term in weights
        self.loss = []
        self.X = np.c_[np.ones((X.shape[0], 1)), X]  # Augment X with bias term

    def _mse(self, y_pred, y_actual):
        return np.mean((y_pred - y_actual) ** 2)

    def _fit(self):
        return np.dot(self.X, self.weights)

    def _compute_gradients(self, y_pred, y_actual):
        return 2 / self.X.shape[0] * np.dot(self.X.T, (y_pred - y_actual))

    def _gradient_descent(self, Y):
        for i in range(self.iterations):
            y_pred = self._fit()
            loss_value = self._mse(y_pred, Y)
            self.loss.append(loss_value)
            grad = self._compute_gradients(y_pred, Y)
            self.weights -= self.alpha * grad

            if i % 10 == 0:
                print("Loss after ", i, "iteration is: ", loss_value)

    def _predict(self, X):
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]  # Augment X with bias term
        return np.dot(X_bias, self.weights)
