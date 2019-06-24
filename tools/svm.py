import numpy as np
from sklearn.svm import LinearSVC
import IPython
import matplotlib.pyplot as plt
from numba import jit

class SVM():


    def __init__(self, alpha, eta, intercept=False):

        self.alpha = alpha
        self.eta = eta
        self.coef_ = None
        self.intercept = intercept

    @jit
    def enforce(self, X, y):
        X = np.array(X)
        y = np.array(y)
        unique = np.unique(y)
        assert len(unique == 2)
        assert 1 in unique
        assert -1 in unique
        assert len(X) == len(y)
        y = y * 2  - 1


    @jit
    def multiple_update(self, X, y, K, T):
        X = np.array(X)
        y = np.array(y)
        X_original = X.copy()
        y_original = y.copy()

        if self.intercept:
            X = np.hstack( ( X, np.ones((X.shape[0], 1)) ) )
        y = y * 2 - 1
        self.enforce(X, y)
 
        d = X.shape[1]
        if self.coef_ is None or T==0:
            self.coef_ = np.zeros(d) #np.random.normal(0, 1, d)
        w = self.coef_
        history = []

        hinge_grads = (X.T * y).T

        for epoch in range(K):
            hinge_grads_copy = hinge_grads.copy()
            scores = y * np.dot(X, w)
            hinge_grads_copy[scores > 1] = 0.0

            hinge_mean = - np.mean(hinge_grads_copy, axis=0)
            reg = w * self.alpha

            w = w - self.eta / (T+1) / K * (hinge_mean + reg) 
            # w = w - self.eta * (hinge_mean + reg) 

        self.coef_ = w
        return history

    @jit
    def update(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        X_original = X.copy()
        y_original = y.copy()

        if self.intercept:
            X = np.hstack( ( X, np.ones((X.shape[0], 1)) ) )
        y = y * 2 - 1
        self.enforce(X, y)
 
        d = X.shape[1]
        if self.coef_ is None or T==0:
            self.coef_ = np.zeros(d) # np.random.normal(0, 1, d)
        w = self.coef_
        history = []

        hinge_grads = (X.T * y).T

        hinge_grads_copy = hinge_grads.copy()
        scores = y * np.dot(X, w)
        hinge_grads_copy[scores > 1] = 0.0

        hinge_mean = - np.mean(hinge_grads_copy, axis=0)
        reg = w * self.alpha

        w = w - self.eta / (T+1) * (hinge_mean + reg) 
        # w = w - self.eta * (hinge_mean + reg) 

        self.coef_ = w
        return history

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

        min_loss = None
        hinge_grads = (X.T * y).T

        for epoch in range(1, epochs):
            w = self.compute_grad(epoch, hinge_grads, w)


            loss = self._loss(X, y, w=w)
            loss_history[epoch-1] = loss
            param_history[epoch-1, :] = w

        self.coef_ = param_history[np.argmin(loss_history)]
        return loss_history, param_history
        

    @jit
    def compute_grad(self, epoch, hinge_grads, w):
        hinge_grads_copy = hinge_grads.copy()
        scores = hinge_grads_copy.dot(w)
        hinge_grads_copy[scores > 1] = 0.0

        hinge_mean = - np.mean(hinge_grads_copy, axis=0)
        reg = w * self.alpha
        w = w - self.eta / np.sqrt(epoch) * (hinge_mean + reg) 
        return w



    @jit
    def _loss(self, X, y, w=None):
        if w is None:
            w = self.coef_

        scores = X.dot(w) * y
        hinge_losses = np.clip(1 - scores, 0, None)
        hinge = np.mean(hinge_losses)
        reg = self.alpha * .5 * np.square(np.linalg.norm(w))

        return hinge + reg


    @jit
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
        

    @jit
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

    @jit
    def predict(self, X):
        X = np.array(X)
        scores = self.decision_function(X)

        yhat = np.zeros(X.shape[0])
        yhat[np.where(scores > 0)] = 1.0

        return yhat.astype(int)

if __name__ == '__main__':

    alpha = .1
    eta = .1
    svm = SVM(alpha, eta)
    X = np.array([[1, 2], [-2, -1], [-10, -1], [2, 2]])
    y = np.array([1, 0, 0, 1])
    history = svm.fit(X, y)
    print(svm.predict(X))
    plt.plot(history)
    plt.show()
    # print(svm.predict(X))

    # print(est.predict(X))

    # print(svm.coef_)
    # print(est.coef_)

    # plt.plot(history)
    # plt.show()

    # IPython.embed()
