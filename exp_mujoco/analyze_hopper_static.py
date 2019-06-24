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
        'beta': (beta_mean, beta_std)
    }
    return results


def plot_results(all_results, i):

    plt.subplot(2, 2, 1 + i * 2)
    plt.title(all_results[0]['title'] + ": Regret", fontsize=22)
    for results in all_results:
        sr_mean, sr_std = results['static_regret']
        diff_mean, diff_std = results['dyn_regret']
        x_axis = np.arange(len(sr_mean))
        p = plt.errorbar(x_axis, sr_mean, yerr=sr_std, label=results['label'] + ': Static Regret', linestyle='--', linewidth=3.0)
        # plt.errorbar(x_axis, diff_mean, yerr=diff_std, label=results['label'], linewidth=3.0)
    plt.ylim(-3, 20)
    plt.tight_layout()
    plt.legend(fontsize=20, loc='upper right')
    if i == 1:
        plt.xlabel('Iterations', fontsize=20)
    plt.ylabel('Regret', fontsize=20)
    # plt.ylim(0, 1)



    plt.subplot(2, 2, i * 2 + 2)
    plt.title(all_results[0]['title'] + ": Distance Traveled", fontsize=22)
    for results in all_results:
        sup_reward_mean, reward_std = results['sup_reward']
        reward_mean, sup_reward_std = results['reward']
        p = plt.errorbar(x_axis, reward_mean, yerr=reward_std, label=results['label'], linewidth=3.0)
    plt.errorbar(x_axis, sup_reward_mean, yerr=sup_reward_std, label='Supervisor', linestyle='--', color="green", linewidth=3.0)
    plt.ylabel('Distance Traveled', fontsize=20)

    if i == 1:
        plt.xlabel('Iterations', fontsize=20)
    plt.tight_layout()
    plt.legend(fontsize=20, loc='upper right', ncol=2)
    plt.ylim(-200, 600)
    # plt.show()




if __name__ == '__main__':
    paths = ['data/mujoco/Hopper/other_frame_skipNone_force_mag5.0', 'data/mujoco/Hopper/reg_other_frame_skipNone_force_mag5.0']
    titles = ['Hopper', 'Hopper']
    plt.figure(figsize=(20, 10))
    for i, (path, title) in enumerate(zip(paths, titles)):
        algs = ['dagger', 'ig', 'mig']
        if i == 1:
            labels = ["DAgger+AOR", "IG+AOR", "MIG+AOR"]
        else:
            labels = ["DAgger", "IG", "MIG"]
        results = []
        for label, alg in zip(labels, algs):
            # alg = 'mig'

            direc = os.path.join(path, alg)
            # direc = os.path.join('data/cartpole', alg)
            prefix = alg
            filepath = os.path.join(direc, prefix) + '.p'
            low_results = aggregate(filepath)
            low_results['label'] = label
            low_results['title'] = title
            results.append(low_results)

        plot_results(results, i)



    plt.savefig('images/hopper_static.pdf')


