import numpy as np
from sklearn.model_selection import train_test_split
import IPython
from sklearn.exceptions import NotFittedError

"""
    sup should have two methods: intended_action() and sample_action()
    which return the intended action and the potentially noisy action respectively.
"""

class Learner():

    def __init__(self, est, sup=None):
        self.X = []
        self.y = []
        self.est = est
        self.one_class_error = None

    def add_data(self, states, actions):
        assert type(states) == list
        assert type(actions) == list
        assert len(states) == len(actions)
        self.X += states
        self.y += actions

    def set_data(self, states, actions):
        assert type(states) == list
        assert type(actions) == list
        assert len(states) == len(actions)
        self.X = states
        self.y = actions

    def set_update(self, states, actions):
        assert type(states) == list
        assert type(states) == list
        assert type(actions) == list
        assert len(states) == len(actions)
        self.X_update = states
        self.y_update = actions

    def clear_data(self):
        self.X = []
        self.y = []

    def train(self, verbose=False):
        X_train, y_train = self.X, self.y
        try:
            self.est.fit(X_train, y_train)
            self.one_class_error = None
        except ValueError:
            self.one_class_error = y_train[0]


        if verbose == True:
            print("Train score: " + str(self.est.score(X_train, y_train)))

    def update(self, T):
        X_train, y_train = self.X_update, self.y_update
        self.est.update(X_train, y_train, T)


    def multiple_update(self, T):
        X_train, y_train = self.X_update, self.y_update
        self.est.multiple_update(X_train, y_train, 20, T)


    def acc(self):
        if self.one_class_error is not None:
            predictions = np.ones(len(self.y)) * self.one_class_error
            return np.mean((predictions == np.array(self.y)).astype(int))
            
        return self.est.score(self.X, self.y)

    def intended_action(self, s):
        if self.one_class_error is not None:
            return self.one_class_error
        try:
            return self.est.predict([s])[0]
        except NotFittedError:
            return np.random.choice([0, 1])

    def sample_action(self, s):
        return self.intended_action(s)


    def decision_function(self, s):
        if self.one_class_error is not None:
            return self.one_class_error
        return self.est.decision_function([s])[0]

