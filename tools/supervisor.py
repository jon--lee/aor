import numpy as np
from expert import tf_util

class Supervisor():

    def __init__(self, act):
        self.act = act

    def sample_action(self, s):
        return self.intended_action(s)

    def intended_action(self, s):
        action = self.act(s[None], stochastic=False)[0]
        return action
        


class Supervisor2():
    def __init__(self, policy_fn, sess):
        self.policy_fn = policy_fn
        self.sess = sess
        with self.sess.as_default():
            tf_util.initialize()

    def sample_action(self, s):
        with self.sess.as_default():
            intended_action = self.policy_fn(s[None,:])[0]
            return intended_action

    def intended_action(self, s):
        return self.sample_action(s)

class Supervisor3():
    def __init__(self, act):
        self.act = act

    def sample_action(self, s):
        return self.intended_action(s)

    def intended_action(self, s):
        action = self.act(False, s)[0]
        return action
