import numpy as np
import scipy.stats
import random
import IPython
from sklearn.metrics import hinge_loss

def F2(env, pi1, pi2, sup, T, num_samples=1):
    losses = []
    for i in range(num_samples):
        # collect trajectory with states visited and actions taken by agent
        tmp_states, _, _, _ = collect_traj(env, pi1, T)
        tmp_actions = np.array([pi2.intended_action(s) for s in tmp_states])
        tmp_scores = np.array([pi2.decision_function(s) for s in tmp_states])
        sup_actions = np.array([sup.intended_action(s) for s in tmp_states])
        n = len(sup_actions)


        hinge = hinge_loss(sup_actions, tmp_scores)
        penalty = pi2.est.alpha * .5 * np.square(np.linalg.norm(pi2.est.coef_))
        print("hinge: " + str(hinge))
        print("penalty: " + str(penalty))
        errors = hinge / n + penalty

        # compute the mean error on that trajectory (may not be T samples since game ends early on failures)
        losses.append(np.mean(errors))

    # compute the mean and sem on averaged losses.
    return stats(losses)

def F(env, pi1, pi2, sup, T, num_samples=1):
    losses = []
    for i in range(num_samples):
        # collect trajectory with states visited and actions taken by agent
        tmp_states, _, _, _ = collect_traj(env, pi1, T)
        tmp_actions = np.array([pi2.intended_action(s) for s in tmp_states])
        sup_actions = np.array([sup.intended_action(s) for s in tmp_states])
        errors = 1.0 - np.mean(sup_actions == tmp_actions)
        # compute the mean error on that trajectory (may not be T samples since game ends early on failures)
        losses.append(np.mean(errors))

    # compute the mean and sem on averaged losses.
    return stats(losses)

def eval_agent_statistics_discrete(env, lnr, sup, T, num_samples=1):
    """
        evaluate loss in the given environment along the agent's distribution
        for T timesteps on num_samples
    """
    losses = []
    for i in range(num_samples):
        # collect trajectory with states visited and actions taken by agent
        tmp_states, _, tmp_actions, _ = collect_traj(env, lnr, T)
        sup_actions = np.array([sup.intended_action(s) for s in tmp_states])
        errors = 1.0 - np.mean(sup_actions == tmp_actions)
        # compute the mean error on that trajectory (may not be T samples since game ends early on failures)
        losses.append(np.mean(errors))

    # compute the mean and sem on averaged losses.
    return stats(losses)

def stats(losses):
    if len(losses) == 1: sem = 0.0
    else: sem = scipy.stats.sem(losses)

    d = {
        'mean': np.mean(losses),
        'sem': sem
    }
    return d

def ste(trial_rewards):
    if trial_rewards.shape[0] == 1:
        return np.zeros(trial_rewards.shape[1])
    return scipy.stats.sem(trial_rewards, axis=0)

def mean(trial_rewards):
    return np.mean(trial_rewards, axis=0)


def mean_sem(trial_data):
    s = ste(trial_data)
    m = mean(trial_data)
    return m, s


def evaluate_lnr_discrete(env, lnr, sup, T):
    stats = eval_agent_statistics_discrete(env, lnr, sup, T, 1)
    return stats['mean']


def collect_traj(env, agent, T, visualize=False):
    """
        agent must have methods: sample_action and intended_action
        Run trajectory on sampled actions
        record states, sampled actions, intended actions and reward
    """
    states = []
    intended_actions = []
    taken_actions = []

    s = env.reset()

    reward = 0.0

    for t in range(T):

        a_intended = agent.intended_action(s)
        a = agent.sample_action(s)
        next_s, r, done, _ = env.step(a)
        reward += r

        states.append(s)
        intended_actions.append(a_intended)
        taken_actions.append(a)

        s = next_s

        if visualize:
            env.render()

        if done:
            break


    return states, intended_actions, taken_actions, reward
