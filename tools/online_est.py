import numpy as np
from sklearn.linear_model import SGDClassifier
import IPython


class OnlineEst():

    def __init__(self, labels):
        alpha = .1
        self.labels = labels
        self.alpha = .1
        self.est = SGDClassifier(loss='hinge', penalty = 'l2', alpha=self.alpha, learning_rates='constant')

    def fit(self, X, y, iters=100):
        try:
            self.est.coef_ = np.zeros(self.est.coef_.shape)
            self.est.intercept_ = np.zeros(self.est.intercept_.shape)
        except:
            pass
        for _ in range(iters):
            res = self.est.partial_fit(X, y, classes=self.labels)
        return res

    def update(self, X, y):
        res = self.est.partial_fit(X, y, classes=self.labels)
        return res

    def predict(self, X):
        return self.est.predict(X)

    def score(self, X, y):
        return self.est.score(X, y)

    def decision_function(self, X):
        return self.est.decision_function(X)

    @property
    def coef_(self):
        return self.est.coef_
    
    @property
    def intercept_(self):
        return self.est.intercept_


if __name__ == '__main__':

    X = np.array([[1, 1], [2, 1], [10, 10], [20, 10]])
    Y = np.array([1, 1, 2, 2])
    clf = SGDClassifier()
    clf.fit(X, Y)
    print clf.predict(X)

    est = OnlineEst(np.arange(6))

    est.fit(X, Y)

    other = SGDClassifier(loss='hinge', penalty = 'l2', alpha=.1)
    other.fit(X, Y)

    IPython.embed()


