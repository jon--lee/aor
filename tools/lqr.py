import numpy as np
import gym
import scipy.linalg
import IPython

env = gym.envs.make('CartPoleContinuous-v0').env

A = np.zeros((4, 4))
B = np.zeros((4, 1))

for i in range(4):
    dd = 1e-7
    state = np.zeros(4)
    state2 = np.zeros(4)
    state2[i] += dd
    control = np.zeros(1)

    env.reset()
    env.state = state
    res, _, _, _ = env.step(control)

    env.reset()
    env.state = state2
    res2, _, _, _ = env.step(control)

    diff = (res2 - res) / dd

    A[:, i] = diff

for j in range(1):

    state = np.zeros(4)
    control = np.zeros(1)
    control2 = np.zeros(1)
    control2[j] += dd

    env.reset()
    env.state = state
    res, _, _, _ = env.step(control)

    env.reset()
    env.state = state
    res2, _, _, _ = env.step(control2)

    diff = (res2 - res) / dd

    B[:, j] = diff


Q = np.diag([1e2, 1e2, 1.0, 1.0])
R = np.identity(1)
P = scipy.linalg.solve_discrete_are(A, B, Q, R)
K = np.linalg.inv(R + B.T.dot(P).dot(B)).dot(B.T.dot(P).dot(A))

def run():
    x = env.reset()
    env.state = np.array([0.0, 0.0, .7, .01])
    for i in range(300):
        env.render()
        u = -K.dot(x).flatten()
        x, _, _, _ = env.step(u)

run()
IPython.embed()


