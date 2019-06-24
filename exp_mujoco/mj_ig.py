import matplotlib.pyplot as plt
import gym
from tools.supervisor import Supervisor
from tools.learner import Learner
from tools import statistics
import IPython
import numpy as np
from tools.lr import LR
from exp_mujoco.mj_dagger import MujocoDagger
from exp import analyze_lambdas
import pickle
import os
import argparse

class MujocoIG(MujocoDagger):
    def __init__(self, param, force_mag, envname, reg):
        MujocoDagger.__init__(self, param, force_mag, envname, reg)
        # self.iters = 200
        self.trials = 1

        # Adaptive reg weighting

        self.alpha = 1.0
        self.eta = np.min([.0001, 1.0/self.alpha/10])
        print(self.eta)

        self.inner_eta = 1.0#1/self.alpha

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
            if len(self.data_states) > 0:
                X = np.array(states)
                y = np.array(tmp_actions)
                print("\t Coef norm: " + str( np.linalg.norm(self.lnr.est.coef_) / (X.shape[1] * y.shape[1]) ))

            self.compute_statistics(iteration, results)

            states, tmp_actions, _, _ = statistics.collect_traj(self.env, self.lnr, self.params['T'])
            i_actions = [self.sup.intended_action(s) for s in states]


            self.data_states += states
            self.data_actions += i_actions

            self.lnr.set_update(states, i_actions)
            self.lnr.update(iteration)
            
            # Adaptive regularization:
            if self.reg and (iteration + 1) % 10  == 0:
                mean_lambda = np.mean(results['lambdas'][-10:] + self.lambda_prior)
                
                mean_ratio = np.mean(np.array(results['opt_costs'][-10:]) / np.array(results['lnr_costs'][-10:]))

                if mean_ratio < .998:
                    next_alpha = mean_lambda * self.lnr.est.alpha
                    # self.lnr.est.alpha = (1 - mean_ratio) * next_alpha + mean_ratio * self.lnr.est.alpha
                    self.lnr.est.alpha = self.t * next_alpha + (1 - self.t) * self.lnr.est.alpha
                    self.lnr.est.eta = np.min([.0001, 1.0/self.lnr.est.alpha/10])

                print("\n\n\t\t Updated alpha: " + str(self.lnr.est.alpha))
                print("\t\t Mean ratio: " + str(mean_ratio))
                print("\t\t Lambda was: " + str(mean_lambda))
                print("\t\t Eta: " + str(self.lnr.est.eta))




        for key in results.keys():
            results[key] = np.array(results[key])

        self.compute_results(results)

        return results





def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frame_skip', type=int, required=False, default=None)
    ap.add_argument('--env', type=str, required=True)
    ap.add_argument('--force_mag', type=float, required=False, default=1.0)
    ap.add_argument('--reg', action='store_true')
    args = vars(ap.parse_args())
    param = args['frame_skip']
    envname = args['env']
    force_mag = args['force_mag']
    reg = args['reg']
    test = MujocoIG(param, force_mag, envname, reg)
    # test = CartpoleIGLambda()
    results = test.run_trials()





if __name__ == '__main__':
    main()
