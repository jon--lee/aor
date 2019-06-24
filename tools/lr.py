import numpy as np
from sklearn.svm import LinearSVC
import IPython
import matplotlib.pyplot as plt
from numba import jit
class LR():


    def __init__(self, alpha, eta, intercept=False, p=None):

        self.alpha = alpha
        self.eta = eta
        self.coef_ = None
        self.p = p
        self.intercept = intercept
        if intercept:
            raise Exception("didn't implement intercept yet")


    def enforce(self, X, y):
        X = np.array(X)
        y = np.array(y)
        assert len(X) == len(y)

    def multiple_update(self, X, y, K, T):
        X = np.array(X)
        y = np.array(y)
        X_original = X.copy()
        y_original = y.copy()

        n = X.shape[0]
        d = X.shape[1]
        p = y.shape[1]
        if self.coef_ is None:
            self.coef_ = np.zeros((d, p))
        w = self.coef_

        var = X.T.dot(X)
        cov = X.T.dot(y)

        for _ in range(K):
            grad = 2.0 / n * X.T.dot(X.dot(w) - y) + self.alpha * w
            # w = w - self.eta / np.sqrt(T+1) * grad / K
            w = w - self.eta * grad / K

        self.coef_ = w


    def update(self, X, y, T):

        X = np.array(X)
        X = np.array(X)
        y = np.array(y)
        X_original = X.copy()
        y_original = y.copy()

        n = X.shape[0]
        d = X.shape[1]
        p = y.shape[1]
        if self.coef_ is None:
            self.coef_ = np.zeros((d, p))
        w = self.coef_


        grad = 2.0 / n * X.T.dot(X.dot(w) - y) + self.alpha * w
        print("\t\t Gradient Norm: " + str(np.linalg.norm(grad) / (d * p)))
        # w = w - self.eta / np.sqrt(T+1) * grad
        w = w - self.eta * grad

        self.coef_ = w

    def fit(self, X, y, w=None, epochs=1000):
        X = np.array(X)
        y = np.array(y)
        X_original = X.copy()
        y_original = y.copy()

        if self.intercept:
            X = np.hstack( ( X, np.ones((X.shape[0], 1)) ) )
 
        d = X.shape[1]
        p = y.shape[1]
        if self.coef_ is None:
            self.coef_ = np.zeros((d, p))
        if not w is None:
            self.coef_ = w
        w = self.coef_


        loss_history = np.zeros(epochs - 1)
        param_history = np.zeros((epochs - 1, d))

        cov = X.T.dot(y)
        self.coef_ = np.linalg.inv(X.T.dot(X) + X.shape[0] * self.alpha / 2.0 * np.identity(d)).dot(cov)
        return loss_history, param_history
        

    def gradient(self, X, y, w):
        X = np.array(X)
        y = np.array(y)
        self.enforce(X, y)
        n = X.shape[0]
        grad = 2.0 / n * X.T.dot(X.dot(w) - y) + self.alpha * w
        return grad

    def _loss(self, X, y, w=None):
        if w is None:
            w = self.coef_

        n = X.shape[0]
        diff = np.sum(np.square(X.dot(w) - y)) / n
        reg = self.alpha * .5 * np.square(np.linalg.norm(w))

        return diff + reg


    def loss(self, X, y, w=None):
        X = np.array(X)
        y = np.array(y)

        X_original = X.copy()
        y_original = y.copy()

        if self.intercept:
            X = np.hstack( ( X, np.ones((X.shape[0], 1)) ) )
        self.enforce(X, y)

        d = X.shape[1]
        p = y.shape[1]
        if self.coef_ is None:
            self.coef_ = np.zeros((d, p))
        if w is None:
            w = self.coef_

        return self._loss(X, y, w=w)
        
    def predict(self, X):
        X = np.array(X)
        if self.coef_ is None:
            return np.zeros((X.shape[0], self.p))
        predictions = X.dot(self.coef_)
        return predictions


if __name__ == '__main__':

    alpha = .1
    eta = .1
    svm = LR(alpha, eta)
    
    n = 50
    x = np.random.uniform(-5, 5, (n, 1))
    y = -3 * x + np.random.normal(0, 4, (n, 1))
    plt.scatter(x.flatten(), y.flatten())

    history = []
    for i in range(1000):
        svm.update(x, y, i)
        history.append(svm.loss(x, y))
    loss1 = history[-1]


    preds = svm.predict(x)
    plt.plot(x, preds)
    plt.show()

    svm2 = LR(alpha, eta)
    svm2.fit(x, y)
    loss2 = svm2.loss(x, y)
    print("gd loss: " + str(loss1))
    print("opt loss: " + str(loss2))
    

    plt.plot(history)
    plt.show()

    # X = np.array([[1, 2], [-2, -1], [-10, -1], [2, 2]])
    # y = np.array([[1, 1], [0, 0], [0, 0], [1, 1]])



    # svm.fit(X, y)
    # predictions = svm.predict(X)
    IPython.embed()



