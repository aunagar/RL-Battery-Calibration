import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import torch
import random
def get_evaluation_rollouts(policy, env, num_of_paths, max_ep_steps, render= True):

    a_bound = env.action_space.high
    paths = []

    for ep in range(num_of_paths):
        s = env.reset()
        path = {'rewards':[],
                'lrewards':[]}
        for step in range(max_ep_steps):
            if render:
                env.render()
            a = policy.choose_action(s, evaluation=True)
            action = a * a_bound
            action = np.clip(action, -a_bound, a_bound)
            s_, r, done, info = env.step(action)
            l_r = info['l_rewards']

            path['rewards'].append(r)
            path['lrewards'].append(l_r)
            s = s_
            if done or step == max_ep_steps-1:
                paths.append(path)
                break
    if len(paths)< num_of_paths:
        print('no paths is acquired')

    return paths


def evaluate_rollouts(paths):
    total_returns = [np.sum(path['rewards']) for path in paths]
    total_lreturns = [np.sum(path['lrewards']) for path in paths]
    episode_lengths = [len(p['rewards']) for p in paths]
    import matplotlib.pyplot as plt
    [plt.plot(np.arange(0, len(path['rewards'])), path['rewards']) for path in paths]
    try:
        diagnostics = OrderedDict((
            ('return-average', np.mean(total_returns)),
            ('return-min', np.min(total_returns)),
            ('return-max', np.max(total_returns)),
            ('return-std', np.std(total_returns)),
            ('lreturn-average', np.mean(total_lreturns)),
            ('lreturn-min', np.min(total_lreturns)),
            ('lreturn-max', np.max(total_lreturns)),
            ('lreturn-std', np.std(total_lreturns)),
            ('episode-length-avg', np.mean(episode_lengths)),
            ('episode-length-min', np.min(episode_lengths)),
            ('episode-length-max', np.max(episode_lengths)),
            ('episode-length-std', np.std(episode_lengths)),
        ))
    except ValueError:
        print('Value error')
    else:
        return diagnostics


def evaluate_training_rollouts(paths):
    data = copy.deepcopy(paths)
    if len(data) < 1:
        return None
    try:
        diagnostics = OrderedDict((
            ('return', np.mean([np.sum(path['rewards']) for path in data])),
            ('length', np.mean([len(p['rewards']) for p in data])),
        ))
    except KeyError:
        return
    [path.pop('rewards') for path in data]
    for key in data[0].keys():
        result = [np.mean(path[key]) for path in data]
        diagnostics.update({key: np.mean(result)})
    # print(diagnostics)
    return diagnostics

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(t.data * (1.0 - tau) + s.data * tau)

def hard_update(target, source):
    target.load_state_dict(source.state_dict())

def stop_grad(network):
    for param in network.parameters():
        param.requires_grad = False

class StateStorage(object):

    def __init__(self):
        self.state_dict = dict()
        self.state_size = None
    
    def update(self, predicted_state, original_state):
        assert len(predicted_state) == len(original_state)

        if not self.state_dict:
            self.state_size = len(predicted_state)
            for i in range(len(predicted_state)):
                self.state_dict['s{}'.format(i)] = {'predicted' : [predicted_state[i]],
                                                    'ground_truth':[original_state[i]]}
        else:
            for i in range(len(predicted_state)):
                self.state_dict['s{}'.format(i)]['predicted'].append(predicted_state[i])
                self.state_dict['s{}'.format(i)]['ground_truth'].append(original_state[i])

    def plot_states(self, outpath = ""):
        assert self.state_size, "Nothing is stored in the state storage yet!"
        plt.style.use('seaborn')
        for i in range(self.state_size):
            predicted_traj = self.state_dict['s{}'.format(i)]['predicted']
            ground_traj = self.state_dict['s{}'.format(i)]['ground_truth']
            x = np.linspace(0, len(predicted_traj)-1, len(predicted_traj))
            fig = plt.figure()
            plt.plot(x, predicted_traj, linestyle = '', color='blue', label='Tracking', marker = 'o', markersize = 1)
            plt.plot(x, ground_traj, color = 'orange', linestyle = '', label='Ground Truth', marker = '.', markersize = 3)
            plt.xlabel('time')
            plt.ylabel('state{}'.format(i))
            plt.legend(loc="upper right", markerscale=3., scatterpoints=1, fontsize=10)
            plt.savefig(outpath + '/state{}_tracking.jpg'.format(i))
        
        
