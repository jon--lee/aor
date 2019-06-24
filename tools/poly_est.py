from sklearn.preprocessing import PolynomialFeatures



class PolyEst():

    def __init__(self, base_est):
        self.est = base_est
        self.poly = PolynomialFeatures(2)

    def fit(self, X, y):
        X = self.poly.fit_transform(X)
        return self.est.fit(X, y)

    def update(self, X, y, T):
        X = self.poly.fit_transform(X)
        return self.est.update(X, y, T)

    def multiple_update(self, X, y, K, T):
        X = self.poly.fit_transform(X)
        return self.est.multiple_update(X, y, K, T)

    def predict(self, X):
        X = self.poly.fit_transform(X)
        return self.est.predict(X)

    def score(self, X, y):
        X = self.poly.fit_transform(X)
        return self.est.score(X, y)

    def decision_function(self, X):
        X = self.poly.fit_transform(X)
        return self.est.decision_function(X)

    # def loss(self, X, y):
    #     X = self.poly.fit_transform(X)
    #     return self.est.loss(X, y)

    @property
    def coef_(self):
        return self.est.coef_

    @property
    def intercept_(self):
        return self.est.intercept_

    def gradient(self, X, y, w=None):
        X = self.poly.fit_transform(X)
        return self.est.gradient(X, y, w=w)


    @property
    def alpha(self):
        return self.est.alpha
    
    def loss(self, X, y, w=None):
        X = self.poly.fit_transform(X)
        return self.est.loss(X, y, w=w)
   
    @property
    def alpha(self):
        return self.est.alpha
    