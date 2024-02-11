import numpy as np

class LinearRegression():

    def __init__(self, verbose = False, n_iterations= 10000, learning_rate = 0.01,
                  regularization = None, regularization_strength = 0):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.regularization_strength = regularization_strength
        self.verbose = verbose


    def _compute_mse_loss(self, y_pred, y_actual):
        return np.mean((y_actual - y_pred)**2)
            
    def fit(self, X,y):
        self.weights = np.zeros((X.shape[1], 1))
        self.bias = 0
        
        for i in range(self.n_iterations):
            y_pred = np.dot(X, self.weights) + self.bias

            # compute loss
            loss = self._compute_mse_loss(y_pred, y)
            if self.verbose:
                print(f"Loss at iteration {i}:{loss}")

            # update weights and biases
            dw = (2/X.shape[0]) * (np.dot(X.T, (y_pred- y)))
            db = (2 * np.mean(y_pred - y))
            
            if self.regularization == 'l1':
                dw = dw + self.regularization_strength * np.sign(self.weights)
            if self.regularization == 'l2':
                dw = dw +  2 * self.regularization_strength * self.weights

            self.weights = self.weights - self.learning_rate *  dw
            self.bias = self.bias - self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
        
    



