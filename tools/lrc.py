import numpy as np
from sklearn.svm import LinearSVC
import IPython
import matplotlib.pyplot as plt
from numba import jit

class LRC():


    def __init__(self, alpha, eta, intercept=False):

        self.alpha = alpha
        self.eta = eta
        self.coef_ = None
        self.intercept = intercept
        if intercept:
            raise Exception("didn't implement intercept yet")


    def enforce(self, X, y):
        X = np.array(X)
        y = np.array(y)
        unique = np.unique(y)
        assert len(unique == 2)
        assert 1 in unique or -1 in unique
        assert len(X) == len(y)

    def multiple_update(self, X, y, K, T):

        X = np.array(X)
        X = np.array(X)
        y = np.array(y)
        X_original = X.copy()
        y_original = y.copy()

        y = y * 2 - 1
        self.enforce(X, y)

        n = X.shape[0]
        d = X.shape[1]
        if self.coef_ is None:
            self.coef_ = np.zeros(d)
        w = self.coef_

        var = X.T.dot(X)
        cov = X.T.dot(y)

        for _ in range(K):
            grad = 2.0 / n * X.T.dot(X.dot(w) - y) + self.alpha * w
            # w = w - self.eta / np.sqrt(T+1) * grad
            w = w - self.eta * grad / K


            # print("\t\tw: " + str(w))
            # print("\t\tgrad: " + str(grad))
            # print("\t\teta: " + str(self.eta))
        self.coef_ = w


    def update(self, X, y, T):

        X = np.array(X)
        X = np.array(X)
        y = np.array(y)
        X_original = X.copy()
        y_original = y.copy()

        y = y * 2 - 1
        self.enforce(X, y)

        n = X.shape[0]
        d = X.shape[1]
        if self.coef_ is None:
            self.coef_ = np.zeros(d)
        w = self.coef_

        grad = 2.0 / n * X.T.dot(X.dot(w) - y) + self.alpha * w
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
        y = y * 2 - 1
        self.enforce(X, y)
 
        d = X.shape[1]
        if self.coef_ is None:
            self.coef_ = np.zeros(d)#np.random.normal(0, 1, d)
        if not w is None:
            self.coef_ = w
        w = self.coef_


        loss_history = np.zeros(epochs - 1)
        param_history = np.zeros((epochs - 1, d))

        cov = X.T.dot(y)
        self.coef_ = np.linalg.inv(X.T.dot(X) + X.shape[0] * self.alpha / 2 * np.identity(d)).dot(cov)
        return loss_history, param_history
        

    def gradient(self, X, y, w):
        X = np.array(X)
        y = np.array(y)
        y = y * 2 - 1
        self.enforce(X, y)
        n = X.shape[0]
        grad = 2.0 / n * X.T.dot(X.dot(w) - y) + self.alpha * w
        return grad

    def _loss(self, X, y, w=None):
        if w is None:
            w = self.coef_

        diff = np.mean(np.square(X.dot(w) - y))
        reg = self.alpha * .5 * np.square(np.linalg.norm(w))

        return diff + reg


    def loss(self, X, y, w=None):
        X = np.array(X)
        y = np.array(y)

        X_original = X.copy()
        y_original = y.copy()

        if self.intercept:
            X = np.hstack( ( X, np.ones((X.shape[0], 1)) ) )
        y = y * 2 - 1
        self.enforce(X, y)

        d = X.shape[1]
        if self.coef_ is None:
            self.coef_ = np.zeros(d)#np.random.normal(0, 1, d)

        if w is None:
            w = self.coef_

        return self._loss(X, y, w=w)
        
    def decision_function(self, X):
        X = np.array(X)
        X_original = X.copy()
        
        if self.intercept:
            X = np.hstack( ( X, np.ones((X.shape[0], 1)) ) )
        d = X.shape[1]
        if self.coef_ is None:
            self.coef_ = np.zeros(d) #np.random.normal(0, 1, d)

        scores = X.dot(self.coef_)
        return scores


    def predict(self, X):
        X = np.array(X)
        scores = self.decision_function(X)

        yhat = np.zeros(X.shape[0])
        yhat[np.where(scores >= 0)] = 1.0

        return yhat.astype(int)

if __name__ == '__main__':

    alpha = .1
    eta = .1
    svm = LRC(alpha, eta)
    X = np.array([[1, 2], [-2, -1], [-10, -1], [2, 2]])
    y = np.array([1, 0, 0, 1])
    history, _ = svm.fit(X, y)
    print(svm.predict(X))
    plt.plot(history)
    plt.show()
    # print(svm.predict(X))

    # print(est.predict(X))

    # print(svm.coef_)
    # print(est.coef_)

    # plt.plot(history)
    # plt.show()

    IPython.embed()
