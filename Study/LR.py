import numpy as np

class LR:
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def predict(self, x):
        y_hat = self.sigmoid(np.dot(x, self.w) + self.b)
        y = np.where(y_hat >= 0.5, 1, 0)
        return y
    
    def cost_function(self, x, y):
        m = x.shape[0]
        y_hat = self.sigmoid(np.dot(x, self.w) + self.b)

        cost = -y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)
        cost = np.sum(cost) / m

        return cost
    
    def compute_gradient(self, x, y):
        n, m = x.shape
        dj_dw = np.zeros((n,))
        dj_db = 0.
    
        y_hat = self.sigmoid(np.dot(x, self.w) + self.b)
        err = y_hat - y

        dj_dw = np.dot(x.T, err) / m
        dj_db = np.sum(err) / m

        return dj_dw, dj_db
    
    def gradient_descent(self, x, y, alpha=0.1, n_iters=10000):
        J_hist = []

        for i in range(n_iters):
            dj_dw, dj_db = self.compute_gradient(x, y)
            
            self.w -= alpha * dj_dw
            self.b -= alpha * dj_db
            J_hist.append(self.cost_function(x, y))

            if (i % 1000) == 0:
               print('Iteration %5d: Cost %0.2e ' % (i, J_hist[-1]))

        return self.w, self.b, J_hist 
    
    def fit(self, x, y, alpha=0.01, n_iters=10000):
        self.w = np.zeros(x.shape[1])
        self.b = 0. 
        w, b, J_hist = self.gradient_descent(x, y, alpha, n_iters)
        return J_hist
