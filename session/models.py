import numpy as np 

# Constant value just used to avoid divide by zeros!
ETA = 1e-10

# Feel free to change any of this! This is just to get you started! 
class MyLogisticRegression():

    def __init__(self, fit_intercept=True, optimizer='simple_gd', learning_rate=0.001,
                 max_iterations=10000, epochs=32, batch_size=32, stopping_criteria=1e-5):

        # Params to get you started
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.fit_intercept = fit_intercept
        
        # Can implement a 'simple_sgd' optimizer, but if you're ahead then there are many other options! 
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.stopping_criteria = stopping_criteria
        self.trained = False
        
        # These can be initialized here, there, or anywhere
        self.loss = None
        self.W = None

    # I've already created the model prediction, loss, and gradient just so you don't have to fumble a bit.
    def _model_pred(self, X):
        return 1 / (1. + np.exp(-(np.dot(X, self.W))))

    def _compute_loss(self, X, y):
        return np.mean(-y * np.log(self._model_pred(X)) - (1 - y) * np.log(1 - self._model_pred(X)))

    def _compute_gradient(self, X, y): 
        return np.dot(X.T, (self._model_pred(X) - y)) / y.size     

    # This will be your function wrapper for using your sigmoid. Is there anythine else you need or might want to add before predicting?
    def predict_probability(self, X):
        # Implement the rest! 
        
        return self._model_pred(X)

    def fit(self, X, y):
        # Implement the rest!
        
        if self.optimizer == 'simple_gd':
            self.simple_gd(X, y)
            print(f"Final loss: {self.loss}")

        print("Model successfully trained!")
        self.trained = True


    def simple_gd(self, X, y):
        # Implement the rest!

            self.W, self.loss = W, loss
    
    ### ADD CODE HERE TOO: feel free to add more functions, implement different forms of Gradient Descent, have fun! 



            



            
