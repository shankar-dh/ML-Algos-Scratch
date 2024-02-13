import numpy as np

class LogisticRegression():
    def __init__(self, verbose = False, learning_rate = 0.01, n_iterations = 1000,
                 regularization = None, lambda_val =0):
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.lambda_val = lambda_val

    def _compute_log_loss(self, y_actual, y_pred):
        """
        Compute the log loss for binary classification.
        
        Parameters:
        - y_actual: numpy array, actual labels (0 or 1)
        - y_pred: numpy array, predicted probabilities
        
        Returns:
        - log_loss: float, computed log loss value
        """
        return -1 * np.mean(y_actual * np.log(y_pred) + (1 - y_actual) * np.log(1 - y_pred))
    
    def _sigmoid(self, value):
        """
        Compute the sigmoid function of the given value.

        Parameters:
        value (float): The input value.

        Returns:
        float: The sigmoid value of the input.
        """
        return 1 / (1 + np.exp(-value))
    
    def fit(self, X, y):
        """
        Fits the logistic regression model to the training data.

        Parameters:
        X (array-like): The input features of shape (n_samples, n_features).
        y (array-like): The target values of shape (n_samples,1).

        Returns:
        None
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features, 1))
        self.bias = 0
        
        for i in range(self.n_iterations):
            y_pred = self._sigmoid(np.dot(X, self.weights) + self.bias)
            
            # Update weights and bias
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            if self.regularization == 'l1':
                dw+= self.lambda_val * np.sign(self.weights)
            if self.regularization == 'l2':
                dw += self.lambda_val * 2 * self.weights

            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db 

            if self.verbose:
                loss = self._compute_log_loss(y, y_pred)
                print(f'loss at iteration {i}: {loss}')

    def predict(self, X):
        """
        Predicts the class labels for the given input data.

        Parameters:
        X (array-like): The input data to be predicted of shape (n_samples, n_features).

        Returns:
        list: The predicted class labels.
        """
        y_pred =  self._sigmoid(np.dot(X, self.weights) + self.bias)
        y_pred_labels = [1 if x > 0.5 else 0 for x in y_pred]
        return y_pred_labels

            


    
