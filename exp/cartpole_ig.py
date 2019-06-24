import matplotlib.pyplot as plt
import gym
from tools.supervisor import Supervisor
from tools.learner import Learner
from tools import statistics
from baselines import deepq
from baselines import deepq
import IPython
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC, SVC
from tools.poly_est import PolyEst
from sklearn.linear_model import SGDClassifier
# from tools.svm import SVM
from tools.lrc import LRC
from exp.cartpole_dagger import CartpoleDagger
from exp import analyze_lambdas
import pickle
import os
import argparse

class CartpoleIG(CartpoleDagger):
    def __init__(self, force_mag, reg):
        CartpoleDagger.__init__(self, force_mag, reg)
        self.iters = 100
        self.trials = 1

        # Adaptive reg weighting
        self.alpha = .1
        self.eta = np.min([.0001, 1/self.alpha])


        self.inner_eta = 1.0

        self.dir = os.path.join(self.base_dir, 'ig')
        self.prefix = 'ig'
        self.path = os.path.join(self.dir, self.prefix)

        self.t = .01

    def run_iters(self):

        results = {
            'lnr_costs': [],
            'opt_costs': [],
            'variations': [],
            'opt_variations': [],
            'param_norms': [],
            'opt_param_norms': [],
            'lambdas': [],
            'lnr_batch_costs': [],
            'opt_batch_costs': [],
            'static_regret': [],
            'rewards': [],
            'betas': [],
            'alphas': [],

        }

        d = self.env.observation_space.shape[0]
        # self.data_states = [np.zeros(d), np.zeros(d)]
        # self.data_actions = [1, 0]
        self.data_states = []
        self.data_actions = []

        for iteration in range(self.iters):
            print("\tIteration: " + str(iteration))
            print("\tData states: " + str(len(self.data_states)))
            print("\tParameters: " + str(self.lnr.est.coef_))
            self.compute_statistics(iteration, results)

            states, tmp_actions, _, _ = statistics.collect_traj(self.env, self.lnr, self.params['T'])
            i_actions = [self.sup.intended_action(s) for s in states]


            self.data_states += states
            self.data_actions += i_actions

            self.lnr.set_update(states, i_actions)
            self.lnr.update(iteration)
            
            # Adaptive regularization:
            if self.reg and (iteration + 1) % 20  == 0:
                mean_lambda = np.mean(results['lambdas'][-10:] + self.lambda_prior)
                next_alpha = mean_lambda * self.lnr.est.alpha
                self.lnr.est.alpha = self.t * next_alpha + (1 - self.t) * self.lnr.est.alpha
                self.lnr.est.eta = np.min([.0001, 1/self.lnr.est.alpha])

                print("\n\n\t\t Updated alpha: " + str(self.lnr.est.alpha))
                print("\t\t Lambda was: " + str(mean_lambda))




        for key in results.keys():
            results[key] = np.array(results[key])

        self.compute_results(results)

        return results


class CartpoleIGLambda(CartpoleIG):

    def run_trials(self):

        title = 'force_mag'
        self.base_dir = 'data/cartpole_lambda_' + title
        self.dir = os.path.join(self.base_dir, 'ig')
        # params = [.001, .01, .1, 1.0, 10.0, 100.0]
        params = [.5, 2.0, 4.0, 8.0, 16.0, 32.0]
        all_results = []
        self.path
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        for param in params:
            self.prologue()
            self.env.env.force_mag = param
            self.path = os.path.join(self.dir, self.prefix) + '_' + title + str(param)
            results = self.run_iters()
            results['param'] = param
            results['title'] = title
            all_results.append(results)

        analyze_lambdas.aggregate_save(self.dir, self.prefix, all_results)
        filepath = os.path.join(self.dir, self.prefix) + '_lambda.p'
        f = open(filepath, 'wb')
        pickle.dump(all_results, f)
        f.close()

        return all_results



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--force_mag', type=float)
    ap.add_argument('--reg', action='store_true')
    args = vars(ap.parse_args())
    force_mag = args['force_mag']
    reg = args['reg']
    test = CartpoleIG(force_mag, reg)
    # test = CartpoleIGLambda()
    results = test.run_trials()





if __name__ == '__main__':
    main()
