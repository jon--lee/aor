from tools import statistics
import IPython
import numpy as np
import matplotlib.pyplot as plt
import os 
import pickle

def aggregate_save(direc, prefix, all_results):
        n = len(all_results)
        d = len(all_results[0]['lnr_costs'])

        var = np.zeros((n, d-1))
        opt_var = np.zeros((n, d-1))
        lamb = np.zeros((n, d-1))

        params = []
        title = None

        for t, result in enumerate(all_results):
            params.append(result['param'])
            title = result['title']
            var[t, :] = result['variations']
            opt_var[t, :] = result['opt_variations']
            lamb[t, :] = result['lambdas']


        var_mean, var_std = statistics.mean_sem(var.T)
        opt_var_mean, opt_var_std = statistics.mean_sem(opt_var.T)
        lamb_max = np.max(lamb, axis=1)
        lamb_mean, lamb_std = statistics.mean_sem(lamb.T)

        # Dynamic Regret
        plt.subplot(311)
        plt.title("Variations")
        plt.errorbar(params, var_mean, yerr=var_std, label='lnr variations')
        plt.errorbar(params, opt_var_mean, yerr=opt_var_std, label='opt variations')
        plt.xscale('log')
        plt.legend()

        plt.subplot(312)
        plt.title("Max Lambda Ratio")
        plt.xscale('log')
        plt.plot(params, lamb_max)

        plt.subplot(313)
        plt.title("Mean Lambda Ratio")
        plt.errorbar(params, lamb_mean, yerr=lamb_std)
        plt.xscale('log')
        plt.tight_layout()
        
        filepath = os.path.join(direc, prefix) + '_lambda.pdf'
        plt.savefig(filepath)
        plt.close()
        plt.cla()
        plt.clf()




def aggregate(filepath):
    f = open(filepath, 'rb')
    all_results = pickle.load(f)
    f.close()

    n = len(all_results)
    d = len(all_results[0]['lnr_costs'])

    var = np.zeros((n, d-1))
    opt_var = np.zeros((n, d-1))
    lamb = np.zeros((n, d-1))

    params = []
    title = None

    for t, result in enumerate(all_results):
        params.append(result['param'])
        title = result['title']
        var[t, :] = result['variations']
        opt_var[t, :] = result['opt_variations']
        lamb[t, :] = result['lambdas']


    var_mean, var_std = statistics.mean_sem(var.T)
    opt_var_mean, opt_var_std = statistics.mean_sem(opt_var.T)
    lamb_max = np.max(lamb, axis=1)
    lamb_mean, lamb_std = statistics.mean_sem(lamb.T)

    results = {
        'var': [var_mean, var_std],
        'opt_var': [opt_var_mean, opt_var_std],
        'lamb': [lamb_max, lamb_mean, lamb_std]
    }

    return params, results

if __name__ == '__main__':
    alg = 'dagger'
    direc = os.path.join('data/cartpole_lambda_force_mag', alg)
    prefix = alg
    filepath = os.path.join(direc, prefix) + '_lambda.p'
    params, dagger_results = aggregate(filepath)

    alg = 'ig'
    direc = os.path.join('data/cartpole_lambda_force_mag', alg)
    prefix = alg
    filepath = os.path.join(direc, prefix) + '_lambda.p'
    params, ig_results = aggregate(filepath)

    alg = 'mig'
    direc = os.path.join('data/cartpole_lambda_force_mag', alg)
    prefix = alg
    filepath = os.path.join(direc, prefix) + '_lambda.p'
    params, mig_results = aggregate(filepath)


    plt.subplot(111)
    plt.title("Mean Lambda Ratio")
    plt.errorbar(params, dagger_results['lamb'][1], yerr=dagger_results['lamb'][2], label='dagger')
    plt.errorbar(params, ig_results['lamb'][1], yerr=ig_results['lamb'][2], label='ig')
    plt.errorbar(params, mig_results['lamb'][1], yerr=mig_results['lamb'][2], label='mig')
    plt.xscale('log')
    plt.tight_layout()
    plt.legend()
    plt.show()

    IPython.embed()













