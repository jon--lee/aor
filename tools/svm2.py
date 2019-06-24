import numpy as np
import matplotlib.pyplot as plt
X = np.array([
    [-2,4,-1],
    [4,1,-1],
    [1, 6, -1],
    [2, 4, -1],
    [6, 2, -1],

])

y = np.array([-1,-1,1,1,1])

def evaluate(w, X, Y):

    scores = 1 - X.dot(w) * Y
    scores[scores<0] = 0.0
    mean_hinge = np.mean(scores)

    reg = .5 * np.square(np.linalg.norm(w))
    return mean_hinge + reg



def svm_sgd(X, Y):

    w = np.zeros(len(X[0]))
    eta = 1
    epochs = 100000
    history = []

    for epoch in range(1,epochs):
        for i, x in enumerate(X):
            if (Y[i]*np.dot(X[i], w)) < 1:
                w = w + eta/epoch * ( (X[i] * Y[i]) + (-2  *(1/epoch)* w) )
            else:
                w = w + eta/epoch * (-2  *(1/epoch)* w)

        history.append(evaluate(w, X, Y))

    return w, history

w, history = svm_sgd(X,y)

import IPython
IPython.embed()
print(w)
