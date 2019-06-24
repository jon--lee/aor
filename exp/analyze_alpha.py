import matplotlib
matplotlib.rc('xtick', labelsize=18)
matplotlib.rc('ytick', labelsize=18)
from tools import statistics
import IPython
import numpy as np
import matplotlib.pyplot as plt
import os 
import pickle

plt.style.use('ggplot')

def aggregate(filepath):
    f = open(filepath, 'rb')
    all_results = pickle.load(f)
    f.close()

    n = len(all_results)
    d = len(all_results[0]['lnr_costs'])
    
    lnr_costs = np.zeros((n, d))
    opt_costs = np.zeros((n, d))
    diff_costs = np.zeros((n, d))
    static_regret = np.zeros((n, d))

    rewards = np.zeros((n, d))
    sup_rewards = np.zeros((n, d))
    variations = np.zeros((n, d - 1))
    opt_variations = np.zeros((n, d - 1))
    lambdas = np.zeros((n, d - 1))
    betas = np.zeros((n, d - 1))
    alphas = np.zeros((n, d))

    for t, result in enumerate(all_results):
        lnr_costs[t, :] = result['lnr_costs']
        opt_costs[t, :] = result['opt_costs']
        # lnr_costs[t, :] = np.cumsum(result['lnr_costs']) / (np.arange(d) + 1)
        # opt_costs[t, :] = np.cumsum(result['opt_costs']) / (np.arange(d) + 1)
        rewards[t, :] = result['rewards']
        sup_rewards[t, :] = result['sup_rewards']
        
        variations[t, :] = result['variations']
        opt_variations[t, :] = result['opt_variations']
        lambdas[t, :] = result['lambdas']
        betas[t, :] = result['betas']
        alphas[t, :] = result['alphas']

        static_regret[t, :] = result['static_regret']


    diff_costs = lnr_costs - opt_costs

    lnr_mean, lnr_std = statistics.mean_sem(lnr_costs)
    opt_mean, opt_std = statistics.mean_sem(opt_costs)
    diff_mean, diff_std = statistics.mean_sem(diff_costs)
    sr_mean, sr_std = statistics.mean_sem(static_regret)
    
    var_mean, var_std = statistics.mean_sem(variations)
    opt_var_mean, opt_var_std = statistics.mean_sem(opt_variations)
    lam_mean, lam_std = statistics.mean_sem(lambdas)
    beta_mean, beta_std = statistics.mean_sem(betas)
    alpha_mean, alpha_std = statistics.mean_sem(alphas)
    alpha_alt_mean, alpha_alt_std = statistics.mean_sem(lambdas * betas)

    reward_mean, reward_std = statistics.mean_sem(rewards)
    sup_reward_mean, sup_reward_std = statistics.mean_sem(sup_rewards)

    x_axis = np.arange(len(lnr_mean))

    results = {
        "dyn_regret": (diff_mean, diff_std),
        "static_regret": (sr_mean, sr_std),
        "reward": (reward_mean, reward_std),
        "sup_reward": (sup_reward_mean, sup_reward_std),
        "var": (var_mean, var_std),
        "opt_var": (opt_var_mean, opt_var_std),
        "lam": (lam_mean, lam_std),
        'beta': (beta_mean, beta_std),
        "alpha": (alpha_mean, alpha_std),
        "alpha_alt": (alpha_alt_mean, alpha_alt_std),
    }
    return results


def plot_results(all_results):


    plt.title(all_results[0]['title'] + ": Alpha", fontsize=22)
    for results in all_results:
        alpha_mean, alpha_std = results['alpha']
        x_axis = np.arange(len(alpha_mean))
        plt.errorbar(x_axis, alpha_mean, yerr=alpha_std, label=results['label'], linewidth=3.0)
    plt.legend(fontsize=20)
    plt.ylabel('Alpha Value', fontsize=20)
    plt.xlabel('Iterations', fontsize=20)
    plt.tight_layout()



if __name__ == '__main__':
    path = 'data/reg_cartpole_force_mag10.0'
    title = 'Hard Cart-Pole'
    plt.figure(figsize=(10, 7))

    algs = ['dagger', 'ig', 'mig']
    labels = ['DAgger+AOR', 'IG+AOR', 'MIG+AOR']

    results = []
    for label, alg, in zip(labels, algs):
        direc = os.path.join(path, alg)
        prefix = alg
        filepath = os.path.join(direc, prefix) + '.p'
        low_results = aggregate(filepath)
        low_results['label'] = label
        low_results['title'] = title
        results.append(low_results)

    plot_results(results)
    plt.savefig('images/cartpole_alpha.pdf')

    # paths = ['data/cartpole_force_mag2.0', 'data/cartpole_force_mag10.0', 'data/reg_cartpole_force_mag10.0']
    # titles = ['Easy Cart-Pole', 'Hard Cart-Pole', 'Hard Cart-Pole']
    # plt.figure(figsize=(18, 15))
    # for i, (path, title) in enumerate(zip(paths, titles)):
    #     algs = ['dagger', 'ig', 'mig']
        
    #     if i == 2:
    #         labels = ["DAgger+AOPR", "IG+AOPR", "MIG+AOPR"]
    #     else:
    #         labels = ["DAgger", "IG", "MIG"]
        
    #     results = []
    #     for label, alg in zip(labels, algs):
    #         # alg = 'mig'

    #         direc = os.path.join(path, alg)
    #         # direc = os.path.join('data/cartpole', alg)
    #         prefix = alg
    #         filepath = os.path.join(direc, prefix) + '.p'
    #         low_results = aggregate(filepath)
    #         low_results['label'] = label
    #         low_results['title'] = title
    #         results.append(low_results)

    #     plot_results(results, i)



    plt.savefig('images/cartpole.pdf')


