import numpy as np 

ETA = 1e-10

class MyLogisticRegression():

    def __init__(self, fit_intercept=True, optimizer='simple_gd', learning_rate=0.001,
                 max_iterations=10000, epochs=32, batch_size=32, stopping_criteria=1e-5):

        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.fit_intercept = fit_intercept
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.stopping_criteria = stopping_criteria
        self.trained = False
        self.loss = None
        self.W = None

    def _model_pred(self, X):
        return 1 / (1. + np.exp(-(np.dot(X, self.W))))

    def _compute_loss(self, X, y):
        return np.mean(-y * np.log(self._model_pred(X)) - (1 - y) * np.log(1 - self._model_pred(X)))

    def _compute_gradient(self, X, y): 
        return np.dot(X.T, (self._model_pred(X) - y)) / y.size

    def _add_intercept_bias(self, X):
        W_bias = np.ones((X.shape[0], 1))
        return np.concatenate((W_bias, X), axis=1)

    def _simple_w_init(self, X):
        return np.zeros((X.shape[1], 1))        

    def predict_probability(self, X):
        if not self.trained:
            print("Must train your model first!")
            return None 
        if self.fit_intercept:
            X = self._add_intercept_bias(X)

        return self._model_pred(X)

    def fit(self, X, y):
        if self.fit_intercept:
            X = self._add_intercept_bias(X)
        if self.optimizer == 'simple_gd':
            self.simple_gd(X, y)
            print(f"Final loss: {self.loss}")
        elif self.optimizer == 'mini_batch_sgd':
            self.mini_batch_sgd(X, y)
            print(f"Final loss: {self.loss}")
        else:
            print(f"Incorrect Optimization Algorithm. Expected 'mini_batch_sgd' or 'simple_gd'"
                  f" but got {self.optimizer}")
        print("Model successfully trained!")
        self.trained = True

    def _simple_weight_update(self, X, y, W, lr):
        grad = self._compute_gradient(X, y)
        W = W - lr * grad
        loss = self._compute_loss(X, y)
        return (W, loss)

    def simple_gd(self, X, y):
        self.W = self._simple_w_init(X)
        for i in np.arange(0, self.max_iterations):
            X_shuffled, y_shuffled = self._shuffle_data(X, y)
            W, loss = self._simple_weight_update(X_shuffled, y_shuffled, self.W, self.learning_rate)
            if i % 100 == 0: 
                print(f"loss: {loss}")
            if np.max(np.abs(self.W - W)) / (max(self.W) + ETA) < self.stopping_criteria:
                break

            self.W, self.loss = W, loss
    
    def mini_batch_sgd(self, X, y):
        self.W = self._simple_w_init(X)
        early_stop = False
        for i in np.arange(0, self.epochs):
            if early_stop:
                break
            X_shuffled, y_shuffled = self._shuffle_data(X, y)
            for (X_batch, y_batch) in self._get_batch_data(X_shuffled, y_shuffled, self.batch_size):
                W, loss = self._simple_weight_update(X_batch, y_batch, self.W, self.learning_rate)
                print(f"loss: {loss}")
                if np.max(np.abs(self.W - W)) / (max(self.W) + ETA) < self.stopping_criteria:
                    early_stop = True
                    break

                self.W, self.loss = W, loss

    def _get_batch_data(self, X, y, batch):
        for index in np.arange(0, X.shape[0], batch):
            yield (X[index:index + batch], y[index:index + batch])

    def _shuffle_data(self, X, y):
        shuffle_index = np.arange(0, X.shape[0])
        np.random.shuffle(shuffle_index)
        X_shuffled, y_shuffled = X[shuffle_index], y[shuffle_index]
        return (X_shuffled, y_shuffled)



            



            
