import matplotlib.pyplot as plt
import gym
from tools.supervisor import Supervisor2
from tools.learner import Learner
from tools import statistics
import IPython
import numpy as np
from tools.lr import LR
import os 
import tensorflow as tf
from expert import load_policy
from exp import analyze_lambdas
import pickle
import argparse
from exp.cartpole_dagger import CartpoleDagger
import time as timer

class MujocoDagger(CartpoleDagger):
    def __init__(self, param, force_mag, envname, reg):
        self.reg = reg
        self.iters = 300
        self.T = 200
        self.trials = 1
        
        self.alpha = 1.0
        self.lambda_prior = list(np.ones(10))
        self.t = .1


        self.eta = 1.0   # does not matter for dagger
        self.inner_eta = self.eta
        self.params = {}    
        self.params['T'] = self.T
        self.params['iters'] = self.iters

        self.sess = tf.Session()
        self.policy = load_policy.load_policy(envname + "-v1.pkl")

        self.base_dir = "data/mujoco/" + str(envname) + "/"
        if self.reg:    self.base_dir = self.base_dir + 'reg_other_frame_skip' + str(param) + "_force_mag" + str(force_mag)
        else:           self.base_dir = self.base_dir + 'other_frame_skip' + str(param) + "_force_mag" + str(force_mag)
        self.dir = os.path.join(self.base_dir, 'dagger')
        self.prefix = 'dagger'
        self.path = os.path.join(self.dir, self.prefix)
        self.param = param
        self.force_mag = force_mag
        self.envname = envname

    def prologue(self):
        self.env = gym.envs.make(self.envname + '-v2')
        self.sup = Supervisor2(self.policy, self.sess)
        p = self.env.env.action_space.shape[0]
        self.lnr = Learner(LR(self.alpha, self.eta, intercept=False, p=p))
        self.alphas = []

        print("Force mag: " + str(self.env.env.force_mag))
        self.env.env.force_mag = self.force_mag
        print("New force mag: " + str(self.env.env.force_mag))

        if self.param is not None:
            print("Frame skip: " + str(self.env.env.frame_skip))
            self.env.env.frame_skip = self.param
            print("New frame skip: " + str(self.env.env.frame_skip))


    def compute_statistics(self, iteration, results):
        self.alphas.append(self.lnr.est.alpha)
        states, tmp_actions, _, reward = statistics.collect_traj(self.env, self.lnr, self.params['T'], False)
        actions = [self.sup.intended_action(s) for s in states]
        d = self.env.observation_space.shape[0]


        est = LR(self.lnr.est.alpha, self.inner_eta, intercept=False)
        lh, ph = est.fit(states, actions)

        lnr_cost = self.lnr.est.loss(states, actions)
        opt_cost = est.loss(states, actions)

        # if lnr_cost < opt_cost:
        #     IPython.embed()

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

        # static_est = LR(self.lnr.est.alpha, self.inner_eta, intercept=False)
        static_est = LR(np.mean(self.alphas), self.inner_eta, intercept=False)
        batch_states = self.data_states + states
        batch_actions = self.data_actions + actions

        lh_batch, ph_batch = static_est.fit(batch_states, batch_actions)
        opt_batch_cost = static_est.loss(batch_states, batch_actions)
        lnr_batch_cost = np.mean(results['lnr_costs'])
        static_regret = lnr_batch_cost - opt_batch_cost

        # if lnr_batch_cost < opt_batch_cost:
        #     IPython.embed()

        print("\tlnr_batch_cost: " + str(lnr_batch_cost))
        print("\topt_batch_cost: " + str(opt_batch_cost))
        print()

        results['lnr_batch_costs'].append(lnr_batch_cost)
        results['opt_batch_costs'].append(opt_batch_cost)
        results['static_regret'].append(static_regret)


        return results


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
            # print("\tData states: " + str(len(self.data_states)))
            # if len(self.data_states) > 0:
            #     X = np.array(states)
            #     y = np.array(tmp_actions)
            #     print("\t Coef norm: " + str( np.linalg.norm(self.lnr.est.coef_) / (X.shape[1] * y.shape[1]) ))

            self.compute_statistics(iteration, results)

            states, tmp_actions, _, _ = statistics.collect_traj(self.env, self.lnr, self.params['T'], False)
            i_actions = [self.sup.intended_action(s) for s in states]


            self.data_states += states
            self.data_actions += i_actions

            self.lnr.set_data(self.data_states, self.data_actions)
            self.lnr.train()


            # Adaptive regularization:
            if self.reg and (iteration + 1) % 10  == 0:
                mean_lambda = np.mean(results['lambdas'][-10:] + self.lambda_prior)
                
                mean_ratio = np.mean(np.array(results['opt_costs'][-10:]) / np.array(results['lnr_costs'][-10:]))

                if mean_ratio < .98:
                    next_alpha = mean_lambda * self.lnr.est.alpha
                    # self.lnr.est.alpha = (1 - mean_ratio) * next_alpha + mean_ratio * self.lnr.est.alpha
                    self.lnr.est.alpha = self.t * next_alpha + (1 - self.t) * self.lnr.est.alpha
                
                print("\n\n\t\t Updated alpha: " + str(self.lnr.est.alpha))
                print("\t\t Mean ratio: " + str(mean_ratio))
                print("\t\t Lambda was: " + str(mean_lambda))



        for key in results.keys():
            results[key] = np.array(results[key])

        IPython.embed()
        self.compute_results(results)

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
        filepath = self.path + '_reward.pdf'
        plt.savefig(filepath)
        plt.close()
        plt.cla()
        plt.clf()


        filepath = self.path + '.p'
        f = open(filepath, 'wb')
        pickle.dump(results, f)
        f.close()




    def run_trials(self):
        self.start_time = timer.time()
        all_results = []
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        # TEsting
        # end testing

        for trial in range(self.trials):
            self.prologue()
            self.path = os.path.join(self.dir, self.prefix) + '_trial' + str(trial)
            results = self.run_iters()
            all_results.append(results)


        self.aggregate(all_results)

        filepath = os.path.join(self.dir, self.prefix) + '.p'
        f = open(filepath, 'wb')
        pickle.dump(all_results, f)
        f.close()

        self.end_time = timer.time()
        print("Total time: " + str(self.end_time - self.start_time))

        return all_results

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

    test = MujocoDagger(param, force_mag, envname, reg)
    results = test.run_trials()





if __name__ == '__main__':
    main()
