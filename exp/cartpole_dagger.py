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
from tools.svm import SVM
from tools.lrc import LRC
import os 
from exp import analyze_lambdas
import pickle
import argparse


class CartpoleDagger():
    def __init__(self, force_mag, reg):
        self.reg = reg
        self.iters = 100
        self.T = 200
        self.trials = 1
        
        self.alpha = 0.1
        self.lambda_prior = list(np.ones(10))


        self.eta = 1.0
        self.inner_eta = self.eta
        self.params = {}    
        self.params['T'] = self.T
        self.params['iters'] = self.iters
        self.act = deepq.load("cartpole_model_alt2.pkl")

        if self.reg:    self.base_dir = 'data/reg_cartpole_force_mag' + str(force_mag)
        else:           self.base_dir = 'data/cartpole_force_mag' + str(force_mag)
        self.dir = os.path.join(self.base_dir, 'dagger')
        self.prefix = 'dagger'
        self.path = os.path.join(self.dir, self.prefix)
        self.force_mag = force_mag

        self.t = .01

    def prologue(self):
        # self.env = gym.envs.make('CartPoleAltRandom-v0')     # Used over mutliple trials
        self.env = gym.envs.make('CartPoleAlt-v0')           # Used for just one trial
        self.sup = Supervisor(self.act)
        self.lnr = Learner(LRC(self.alpha, self.eta, intercept=False))
        print(self.env.env.force_mag)
        
        self.env.env.force_mag = self.force_mag

    def run_trials(self):
        all_results = []
        # Used for multiple trials with random initial states
        # init_states = np.load("data/init_states.npy")
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        for trial in range(self.trials):
            self.prologue()
            # print("Init state: " + str(self.env.env.init_state))
            # self.env.env.init_state = init_states[trial, :]
            # print("Setting init state: " + str(self.env.env.init_state))
            self.path = os.path.join(self.dir, self.prefix) + '_trial' + str(trial)
            results = self.run_iters()
            all_results.append(results)


        self.aggregate(all_results)

        filepath = os.path.join(self.dir, self.prefix) + '.p'
        f = open(filepath, 'wb')
        pickle.dump(all_results, f)
        f.close()

        return all_results


    def aggregate(self, all_results):
        n = len(all_results)
        d = self.iters
        
        lnr_costs = np.zeros((n, d))
        opt_costs = np.zeros((n, d))
        diff_costs = np.zeros((n, d))

        lnr_batch_costs = np.zeros((n, d))
        opt_batch_costs = np.zeros((n, d))
        static_regret = np.zeros((n, d))

        for t, result in enumerate(all_results):
            lnr_costs[t, :] = result['lnr_costs']
            opt_costs[t, :] = result['opt_costs']
            diff_costs[t, :] = result['lnr_costs'] - result['opt_costs']
            lnr_batch_costs[t, :] = result['lnr_batch_costs']
            opt_batch_costs[t, :] = result['opt_batch_costs']
            static_regret[t, :] = result['static_regret']


        lnr_mean, lnr_std = statistics.mean_sem(lnr_costs)
        opt_mean, opt_std = statistics.mean_sem(opt_costs)
        diff_mean, diff_std = statistics.mean_sem(diff_costs)

        lnr_batch_mean, lnr_batch_std = statistics.mean_sem(lnr_batch_costs)
        opt_batch_mean, opt_batch_std = statistics.mean_sem(opt_batch_costs)
        static_regret_mean, static_regret_sem = statistics.mean_sem(static_regret)

        x_axis = np.arange(len(lnr_mean))

        # Dynamic Regret
        plt.subplot(211)
        plt.title("Actual loss")
        plt.errorbar(x_axis, lnr_mean, yerr=lnr_std, label='lnr costs')
        plt.errorbar(x_axis, opt_mean, yerr=opt_std, label='opt costs')
        plt.legend()

        plt.subplot(212)
        plt.title("Difference")
        plt.errorbar(x_axis, diff_mean, yerr=diff_std)
        plt.tight_layout()
        
        filepath = os.path.join(self.dir, self.prefix) + '.pdf'
        plt.savefig(filepath)
        plt.close()
        plt.cla()
        plt.clf()


        # Static Regret
        plt.subplot(211)
        plt.title("Batch loss")
        plt.errorbar(x_axis, lnr_batch_mean, yerr=lnr_std, label='lnr costs')
        plt.errorbar(x_axis, opt_batch_mean, yerr=opt_std, label='opt costs')
        plt.legend()

        plt.subplot(212)
        plt.title("Static Regret")
        plt.errorbar(x_axis, static_regret_mean, yerr=diff_std)
        plt.tight_layout()
        
        filepath = os.path.join(self.dir, self.prefix) + '_batch.pdf'
        plt.savefig(filepath)
        plt.close()
        plt.cla()
        plt.clf()






    def compute_statistics(self, iteration, results):

        states, tmp_actions, _, reward = statistics.collect_traj(self.env, self.lnr, self.params['T'], False)
        actions = [self.sup.intended_action(s) for s in states]
        d = self.env.observation_space.shape[0]
        # states += [np.zeros(d), np.zeros(d)]
        # actions += [1, 0]


        est = LRC(self.lnr.est.alpha, self.inner_eta, intercept=False)
        lh, ph = est.fit(states, actions)

        lnr_cost = self.lnr.est.loss(states, actions)
        opt_cost = est.loss(states, actions)

        print("\tlnr_cost: " + str(lnr_cost))
        print("\topt_cost: " + str(opt_cost))


        results['lnr_costs'].append(lnr_cost)
        results['opt_costs'].append(opt_cost)
        results['rewards'].append(reward)
        results['alphas'].append(self.lnr.est.alpha)

        curr_coef_ = self.lnr.est.coef_.copy()
        curr_opt_coef_ = est.coef_.copy()

        results['param_norms'].append(np.linalg.norm(curr_coef_))
        results['opt_param_norms'].append(np.linalg.norm(curr_opt_coef_))

        if not iteration is 0:
        
            variation = np.linalg.norm(self.last_coef_ - curr_coef_)
            opt_variation = np.linalg.norm(self.last_opt_coef_ - curr_opt_coef_)

            last_gradient = est.gradient(self.last_states, self.last_actions, curr_coef_)
            curr_gradient = est.gradient(states, actions, curr_coef_)
            beta = np.linalg.norm(last_gradient - curr_gradient) / variation

            results['variations'].append(variation)
            results['opt_variations'].append(opt_variation)
            results['lambdas'].append(opt_variation / variation)
            results['betas'].append(beta)



        self.last_coef_ = curr_coef_.copy()
        self.last_opt_coef_ = curr_opt_coef_.copy()
        self.last_states = states
        self.last_actions = actions

        static_est = LRC(self.lnr.est.alpha, self.inner_eta, intercept=False)
        batch_states = self.data_states + states
        batch_actions = self.data_actions + actions

        lh_batch, ph_batch = static_est.fit(batch_states, batch_actions)
        opt_batch_cost = static_est.loss(batch_states, batch_actions)
        lnr_batch_cost = np.mean(results['lnr_costs'])
        static_regret = lnr_batch_cost - opt_batch_cost

        print("\tlnr_batch_cost: " + str(lnr_batch_cost))
        print("\topt_batch_cost: " + str(opt_batch_cost))
        print()

        results['lnr_batch_costs'].append(lnr_batch_cost)
        results['opt_batch_costs'].append(opt_batch_cost)
        results['static_regret'].append(static_regret)

        return results


    def compute_results(self, results):

        _, _, _, sup_reward = statistics.collect_traj(self.env, self.sup, self.params['T'], False)
        results['sup_rewards'] = [sup_reward] * len(results['rewards'])

        # DYNAMIC REGRET
        plt.subplot(211)
        plt.title("Actual loss")
        plt.plot(results['lnr_costs'], label='lnr costs')
        plt.plot(results['opt_costs'], label='opt costs')
        plt.legend()


        difference = results['lnr_costs'] - results['opt_costs']
        plt.subplot(212)
        plt.title("Difference")
        plt.plot(difference)
        plt.tight_layout()
        
        filepath = self.path + '.pdf'
        plt.savefig(filepath)
        plt.close()
        plt.cla()
        plt.clf()

        # STATIC REGRET 
        plt.subplot(211)
        plt.title("Batch costs")
        plt.plot(results['lnr_batch_costs'], label='lnr costs')
        plt.plot(results['opt_batch_costs'], label='opt costs')
        plt.legend()


        plt.subplot(212)
        plt.title("Static regret (lnr batch - opt batch)")
        plt.plot(results['static_regret'])
        plt.tight_layout()
        
        filepath = self.path + '_batch.pdf'
        plt.savefig(filepath)
        plt.close()
        plt.cla()
        plt.clf()


        plt.subplot(111)
        plt.title("Rewards")
        plt.plot(results['rewards'], label='Learner rewards')
        plt.plot(results['sup_rewards'], label='Supervisor Rewards')
        plt.legend()
        plt.ylim(0, 20)
        filepath = self.path + '_reward.pdf'
        plt.savefig(filepath)
        plt.close()
        plt.cla()
        plt.clf()


        filepath = self.path + '.p'
        f = open(filepath, 'wb')
        pickle.dump(results, f)
        f.close()


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

            self.compute_statistics(iteration, results)

            states, tmp_actions, _, _ = statistics.collect_traj(self.env, self.lnr, self.params['T'], False)
            i_actions = [self.sup.intended_action(s) for s in states]


            self.data_states += states
            self.data_actions += i_actions

            self.lnr.set_data(self.data_states, self.data_actions)
            self.lnr.train()


            # Adaptive regularization:
            if self.reg and (iteration + 1) % 10  == 0:
                # mean_lambda = np.mean(results['lambdas'][-10:] + self.lambda_prior)
                mean_lambda = np.mean(results['lambdas'][-10:])
                next_alpha = mean_lambda * self.lnr.est.alpha
                self.lnr.est.alpha = self.t * next_alpha + (1 - self.t) * self.lnr.est.alpha
                print("\n\n\t\t Updated alpha: " + str(self.lnr.est.alpha))
                print("\t\t Lambda was: " + str(mean_lambda))


        for key in results.keys():
            results[key] = np.array(results[key])

        self.compute_results(results)

        return results


class CartpoleDaggerLambda(CartpoleDagger):

    def run_trials(self):

        title = 'force_mag'
        self.base_dir = 'data/cartpole_lambda_' + title
        self.dir = os.path.join(self.base_dir, 'dagger')
        # params = [.001, .01, .1, 1.0, 10.0, 100.0]
        # params = [.001, .1, 1.0]
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

    test = CartpoleDagger(force_mag, reg)
    # test = CartpoleDaggerLambda()
    results = test.run_trials()





if __name__ == '__main__':
    main()
