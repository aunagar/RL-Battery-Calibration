# basic imports
import os
import sys
import time
import logger

# library imports
import torch
import h5py
import torch.nn as nn
import numpy as np
from collections import OrderedDict, deque
from copy import deepcopy
import scipy.io as sio
from sklearn import preprocessing
from tqdm import tqdm
import matplotlib.pyplot as plt

# module imports
from .utils import set_seed, hard_update, soft_update, stop_grad, evaluate_training_rollouts
from variant import *
from pool.pool import Pool
from train_eval import training_evaluation

# Fixed Parameters
# SCALE_DIAG_MIN_MAX = (-20,-2.5)
SCALE_DIAG_MIN_MAX = (-20,2)
SCALE_lambda_MIN_MAX = (0, 1)
SCALE_beta_MIN_MAX = (0, 1)
SCALE_alpha_MIN_MAX = (0.01, 1)

# Activation mapping
str_to_activation = {
    'elu': nn.ELU(),
    'hardshrink': nn.Hardshrink(),
    'hardtanh': nn.Hardtanh(),
    'leakyrelu': nn.LeakyReLU(),
    'logsigmoid': nn.LogSigmoid(),
    'prelu': nn.PReLU(),
    'relu': nn.ReLU(),
    'relu6': nn.ReLU6(),
    'rrelu': nn.RReLU(),
    'selu': nn.SELU(),
    'sigmoid': nn.Sigmoid(),
    'softplus': nn.Softplus(),
    'logsoftmax': nn.LogSoftmax(),
    'softshrink': nn.Softshrink(),
    'softsign': nn.Softsign(),
    'tanh': nn.Tanh(),
    'tanhshrink': nn.Tanhshrink(),
    'softmin': nn.Softmin(),
    'softmax': nn.Softmax(dim=1),
    'none': None
    }

def get_data():
    process = VARIANT['dataset_name']
    data = np.load('data/' + process, allow_pickle = True)
    state_orig = data['X']
    X_orig = data['Z']
    W_orig = data['U']
    T_orig = data['theta']

    state = [s[:-1,:] for s in state_orig]
    X = [x[:-1,:] for x in X_orig]
    X_ = [x[1:, :] for x in X_orig]
    W = [w[:-1,:] for w in W_orig]
    T = [t.reshape(-1,1)[:-1,:] for t in T_orig]
    train_frac = 1.0
    data_stack = [np.concatenate((X[i], W[i], X_[i], T[i], state[i]), axis = -1) for i in range(int(train_frac*len(X)))]
    return data_stack

def initialize_weights(initializer):
    def initialize(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            print("Initializing...")
            initializer(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.0)
    return initialize

class BaseNetwork(torch.nn.Module):

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

class Actor(BaseNetwork):

    def __init__(self, input_dim, output_dim, n_layers, layer_sizes, hidden_activation = "relu",
                output_activation = None, bias = True):
        super(Actor, self).__init__()
        assert n_layers == len(layer_sizes), "length of layer_sizes should matches n_layers"
        model = []
        prev_dim = input_dim
        
        for h in layer_sizes:
            model.append(nn.Linear(prev_dim, h, bias = bias))
            model.append(str_to_activation[hidden_activation])
            prev_dim = h
        
        # output layer
        model.append(nn.Linear(prev_dim, output_dim*2, bias = bias))
        self.model = nn.Sequential(*model).apply(initialize_weights(torch.nn.init.kaiming_uniform_))

    def forward(self, states):

        mu, log_sigma = torch.chunk(self.model(states.float()), 2, dim = -1)
        log_sigma = torch.clamp(log_sigma, min = SCALE_DIAG_MIN_MAX[0], max = SCALE_DIAG_MIN_MAX[1])

        sigma = log_sigma.exp()

        distr = torch.distributions.Normal(mu,sigma)

        raw_actions = distr.rsample()
        clipped_actions = torch.tanh(raw_actions)

        # log probs 
        log_probs = distr.log_prob(raw_actions) - torch.log(1 - clipped_actions.pow(2) + 1e-6)
        log_probs = log_probs.sum(1, keepdim = True)
        #clipped mu
        clipped_mu = torch.tanh(mu)

        return clipped_actions, clipped_mu, log_probs, distr


    # def forward(self, states):

    #     encoder = self.model(states.float())
    #     mu = nn.Linear(self.prev_dim, self.output_dim)(encoder)
    #     log_sigma = nn.Linear(self.prev_dim, self.output_dim)(encoder)
    #     log_sigma = torch.clamp(log_sigma, min = SCALE_DIAG_MIN_MAX[0], max = SCALE_DIAG_MIN_MAX[1])
    #     # print("mu = ", mu, " sigma = ", log_sigma)
    #     sigma = log_sigma.exp()
    #     base_distr = torch.distributions.multivariate_normal.MultivariateNormal(
    #                     loc = torch.zeros(self.output_dim),
    #                     covariance_matrix = torch.diag(torch.ones(self.output_dim)))

    #     epsilon = base_distr.rsample()
    #     affine_transform = torch.distributions.transforms.AffineTransform(loc = mu, scale = sigma)
    #     tanh_transform = torch.distributions.transforms.TanhTransform()

    #     raw_action = mu + epsilon*sigma
    #     clipped_action = torch.tanh(raw_action)
    #     orig_distr = torch.distributions.multivariate_normal.MultivariateNormal(
    #                     loc = mu,
    #                     covariance_matrix = torch.diag(sigma).reshape(-1,1))
    #     transformed_distr= torch.distributions.transformed_distribution.TransformedDistribution(
    #         base_distr, [affine_transform, tanh_transform]
    #     )
    #     clipped_mu = torch.tanh(mu)
    #     log_pis = transformed_distr.log_prob(clipped_action)
    #     return clipped_action, clipped_mu, log_pis, orig_distr

class LyapunovCritic(BaseNetwork):
    def __init__(self, state_dim, action_dim, output_dim, n_layers, layer_sizes, hidden_activation = "relu",
                output_activation = None, bias = True):
        super(LyapunovCritic, self).__init__()
        assert n_layers == len(layer_sizes), "length of layer_sizes should matches n_layers"
        model = []
        self.ll1 = nn.Linear(state_dim, layer_sizes[0], bias = False)
        self.ll2 = nn.Linear(action_dim, layer_sizes[0], bias = False)
        self.bias = torch.autograd.Variable(torch.zeros(1, layer_sizes[0]), requires_grad = True)
        self.ll1.apply(initialize_weights(nn.init.kaiming_normal_))
        self.ll2.apply(initialize_weights(nn.init.kaiming_normal_))
        prev_dim = layer_sizes[0]
        self.hidden_activation = hidden_activation
        for h in layer_sizes[1:]:
            model.append(nn.Linear(prev_dim, h, bias = bias))
            model.append(str_to_activation[hidden_activation])
            prev_dim = h

        self.model = nn.Sequential(*model).apply(initialize_weights(nn.init.kaiming_uniform_))
    
    def forward(self, states, actions):
        s = self.ll1(states.float())
        a = self.ll2(actions.float())
        x = str_to_activation[self.hidden_activation](s + a + self.bias)
        x = self.model(x)

        output = torch.unsqueeze(torch.sum(x**2, dim=1), dim=1)

        return output

class CAC(object):
    def __init__(self,
                 a_dim,
                 s_dim,
                 variant,
                 action_prior = 'uniform',
                 max_global_steps = 100000
                 ):
        """
        a_dim : dimension of action space
        s_dim: state space dimension
        variant: dictionary containing parameters for the algorithms
        """
        ###############################  Model parameters  ####################################
        set_seed(variant['seed'])
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(input_dim= s_dim, output_dim= a_dim, n_layers = 3,
                                    layer_sizes = [256,256,256], hidden_activation="leakyrelu").to(self.device)
        self.actor_target = Actor(input_dim= s_dim, output_dim= a_dim, n_layers = 3,
                                    layer_sizes = [256,256,256], hidden_activation="leakyrelu").to(self.device).eval()
        self.critic = LyapunovCritic(state_dim = s_dim, action_dim = a_dim, output_dim = None,
                                        n_layers = 2, layer_sizes=[256,256], hidden_activation="leakyrelu").to(self.device)
        self.critic_target = LyapunovCritic(state_dim=s_dim, action_dim=a_dim, output_dim=None, 
                                            n_layers = 2, layer_sizes=[256, 256],
                                            hidden_activation="leakyrelu").to(self.device).eval()
        
        # copy parameters of the learning network to the target network
        hard_update(self.critic_target, self.critic)
        hard_update(self.actor_target, self.actor)
        # disable gradient calculations of the target network
        stop_grad(self.critic_target)
        stop_grad(self.actor_target)
        # self.memory_capacity = variant['memory_capacity']
        
        ################################ parameters for training ###############################
        self.batch_size = variant['batch_size'] # batch size for learning the actor
        self.gamma = variant['gamma'] # discount factor
        self.tau = variant['tau'] # smoothing parameter for the weight updates   
        self.approx_value = True if 'approx_value' not in variant.keys() else variant['approx_value']
        self._action_prior = action_prior # prior over action space
        s_dim = s_dim * (variant['history_horizon']+1)
        self.a_dim, self.s_dim, = a_dim, s_dim
        self.history_horizon = variant['history_horizon'] # horizon to consider for the history
        self.working_memory = deque(maxlen=variant['history_horizon']+1) # memory to store history
        target_entropy = variant['target_entropy']
        if target_entropy is None:
            self.target_entropy = -self.a_dim  #lower bound of the policy entropy
        else:
            self.target_entropy = target_entropy
        self.target_variance = 0.0
        self.finite_horizon = variant['finite_horizon']
        self.soft_predict_horizon = variant['soft_predict_horizon']
        self.use_lyapunov = variant['use_lyapunov']
        self.adaptive_alpha = variant['adaptive_alpha']
        self.adaptive_beta = variant['adaptive_beta'] if 'adaptive_beta' in variant.keys() else False
        self.time_near = variant['Time_near']
        self.max_global_steps = max_global_steps
        self.LR_A = variant['lr_a']
        self.LR_L = variant['lr_l']
        self.LR_lag = self.LR_A/10
        self.alpha3 = variant['alpha3']

        labda = variant['labda'] # formula (12) in the paper 
        alpha = variant['alpha'] # entropy temperature (beta in the paper)
        beta = variant['beta'] # constraint error weight

        self.log_labda = torch.log(torch.tensor([labda], device = self.device))
        self.log_alpha = torch.log(torch.tensor([alpha], device = self.device))  # Entropy Temperature
        self.log_beta = torch.log(torch.tensor([beta], device = self.device))
        self.log_alpha.requires_grad = True
        self.log_beta.requires_grad = True
        self.log_labda.requires_grad = True
        # The update is in log space
        self.labda = torch.clamp(torch.exp(self.log_labda), min = SCALE_lambda_MIN_MAX[0],
                                max = SCALE_lambda_MIN_MAX[1])
        self.alpha = torch.exp(self.log_alpha)
        self.beta = torch.clamp(torch.exp(self.log_beta),min = SCALE_beta_MIN_MAX[0],
                                max = SCALE_beta_MIN_MAX[1])

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr = self.LR_A)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr = self.LR_L)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr = self.LR_A)
        self.labda_optim = torch.optim.Adam([self.log_labda], lr = self.LR_lag)
        self.beta_optim = torch.optim.Adam([self.log_beta], lr = 0.01)

        step_fn = lambda i : 1.0 - (i - 1.)/self.max_global_steps
        self.actor_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(self.actor_optim, lr_lambda = step_fn)
        self.critic_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(self.critic_optim, lr_lambda = step_fn)
        self.alpha_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(self.alpha_optim, lr_lambda = step_fn)
        self.labda_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(self.labda_optim, lr_lambda = step_fn)
        self.beta_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(self.beta_optim, lr_lambda = step_fn)

        self.actor.float()
        self.critic.float()

    def act(self, s, evaluation = False):
        a, deterministic_a, _, _ = self.actor(s)
        if evaluation is True:
            return deterministic_a
        else:
            return a

    def learn(self, batch):

        # update target networks
        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.actor_target, self.actor, self.tau)

        bs = torch.tensor(batch['s'], dtype = torch.float).to(self.device)  # state
        ba = torch.tensor(batch['a'], dtype = torch.float).to(self.device)  # action
        br = torch.tensor(batch['r'], dtype = torch.float).to(self.device)  # reward
        bterminal = torch.tensor(batch['terminal'], dtype = torch.float).to(self.device)
        bs_ = torch.tensor(batch['s_'], dtype = torch.float).to(self.device)  # next state
        b_s = torch.tensor(batch['_s'], dtype = torch.float).to(self.device) # prev state
        bv = None
        b_r_ = None
        # print(bs)
        alpha_loss = None 
        beta_loss = None

        # # alpha learning
        # if self.adaptive_alpha:
        #     self.alpha_optim.zero_grad()
        #     alpha_loss = self.get_alpha_loss(bs, self.target_entropy)
        #     alpha_loss.backward(retain_graph = False)
        #     self.alpha_optim.step()
        
        # # beta learning
        # self.beta_optim.zero_grad()        
        # beta_loss = self.get_beta_loss(b_s)
        # if self.adaptive_beta:
        #     beta_loss.backward(retain_graph = False)
        #     self.beta_optim.step()
        # else:
        #     self.beta_optim.zero_grad()
        
        # # labda learning
        # self.labda_optim.zero_grad()
        # labda_loss = self.get_labda_loss(br, bs, bs_, ba)
        # # print("labda loss = ", labda_loss)
        # labda_loss.backward(retain_graph = False)
        # self.labda_optim.step()
        
        # actor lerning
        self.actor_optim.zero_grad()
        actor_loss = self.get_actor_loss(bs, bs_, ba, br)
        # print("actor loss = ", actor_loss)
        actor_loss.backward(retain_graph = True)
        self.actor_optim.step()

        if self.finite_horizon:
            bv = torch.tensor(batch['value'])
            b_r_ = torch.tensor(batch['r_N_'])
        # lyapunov learning
        self.critic_optim.zero_grad()
        # print(bs_)
        critic_loss = self.get_lyapunov_loss(bs, bs_, ba, br, b_r_, bv, bterminal)
        critic_loss.backward()
        self.critic_optim.step()

        # alpha learning
        if self.adaptive_alpha:
            self.alpha_optim.zero_grad()
            alpha_loss = self.get_alpha_loss(bs, self.target_entropy)
            alpha_loss.backward(retain_graph = False)
            self.alpha_optim.step()
        
        # labda learning
        self.labda_optim.zero_grad()
        labda_loss = self.get_labda_loss(br, bs, bs_, ba)
        # print("labda loss = ", labda_loss)
        labda_loss.backward(retain_graph = False)
        self.labda_optim.step()

        return alpha_loss, beta_loss, labda_loss, actor_loss, critic_loss

    def get_alpha_loss(self, s, target_entropy):

        # with torch.no_grad():
        #     _, self.deterministic_a,self.log_pis, _ = self.actor_target(s)
        log_pis = self.log_pis
        # self.a, self.deterministic_a, self.log_pis, _ = self.actor(s)
        # print(self.a)
        
        intermediate = log_pis + target_entropy
        return -torch.mean(self.log_alpha*intermediate.detach())
    
    def get_labda_loss(self, r, s, s_, a):
        # with torch.no_grad():
        #     l = self.critic(s, a)
        #     lya_a_, _, _, _ = self.actor_target(s_)
        #     self.l_ = self.critic_target(s_, lya_a_)
        lyapunov_loss = torch.mean(self.l_.detach() - self.l + self.alpha3 * r)
        return -torch.mean(self.log_labda*lyapunov_loss)

    def get_beta_loss(self, _s):
        with torch.no_grad():
            _, _deterministic_a, _ ,_ = self.actor_target(_s)
        self.l_action = torch.mean(torch.norm(_deterministic_a.detach() - self.deterministic_a, dim = 1))
        with torch.no_grad():
            intermediate = (self.l_action - 0.02).detach()
        return -torch.mean(self.log_beta*intermediate)
    
    def get_actor_loss(self, s, s_, a, r):
        if self._action_prior == 'normal':
            policy_prior = torch.distributions.MultivariateNormal(
                loc=torch.zeros(self.a_dim),
                covariance_matrix=torch.diag(torch.ones(self.a_dim)))
            policy_prior_log_probs = policy_prior.log_prob(self.a)
        elif self._action_prior == 'uniform':
            policy_prior_log_probs = 0.0
        
        # only actor weights are updated!
        _, self.deterministic_a, self.log_pis, _ = self.actor(s)
        with torch.no_grad():
            self.l = self.critic(s, a)
            lya_a_, _, _, _ = self.actor(s_)
            self.l_ = self.critic(s_, lya_a_)
        self.lyapunov_loss = torch.mean(self.l_.detach() - self.l + self.alpha3*r)
        a_loss = self.labda * self.lyapunov_loss + self.alpha *torch.mean(self.log_pis) - policy_prior_log_probs
        return a_loss

    def get_lyapunov_loss(self, s, s_, a, r, r_n_ = None, v = None, terminal = 0.):
        with torch.no_grad():
            a_, _, _,_ = self.actor_target(s_)
            l_ = self.critic_target(s_, a_)
        l = self.critic(s, a)
        if self.approx_value:
            if self.finite_horizon:
                if self.soft_predict_horizon:
                    l_target = r - r_n_ + l_.detach()
                else:
                    l_target = v
            else:
                l_target = r + self.gamma * (1 - terminal)*l_.detach()  # Lyapunov critic - self.alpha * next_log_pis
        else:
            l_target = r
        mse_loss = nn.MSELoss()
        l_loss = mse_loss(l, l_target)

        return l_loss

    def save_result(self, path):
        if not os.path.exists(path + "/policy/"):
            os.mkdir(path + "/policy/")
        self.actor_target.save(path + "/policy/actor_target.pth")
        self.critic_target.save(path + "/policy/critic_target.pth")
        self.actor.save(path + "/policy/actor.pth")
        self.critic.save(path + "/policy/critic.pth")
        print("Save to path: ", path + "/policy/")

    def restore(self, path):
        result_path = path
        if not os.path.exists(result_path):
            raise IOError("Results path ", result_path, " does not contain anything to load")
        self.actor_target.load(result_path + "/actor_target.pth")
        self.critic_target.load(result_path + "/critic_target.pth")
        self.actor.load(result_path + "/actor.pth")
        self.critic.load(result_path + "/critic.pth")
        success_load = True
        print("Load successful, model file from ", result_path)
        print("#########################################################")
        return success_load

    def scheduler_step(self):
        self.alpha_scheduler.step()
        self.beta_scheduler.step()
        self.labda_scheduler.step()
        self.actor_scheduler.step()
        self.critic_scheduler.step()

def train(variant):
    Min_cost = 1000000

    data_trajectories = get_data() # get data (X, W, X_, theta, state)
    env_name = variant['env_name'] # choose your environment
    env = get_env_from_name(env_name)

    env_params = variant['env_params']

    max_episodes = env_params['max_episodes'] # maximum episodes for RL training
    max_ep_steps = env_params['max_ep_steps'] # number of maximum steps in each episode
    max_global_steps = env_params['max_global_steps']
    store_last_n_paths = variant['store_last_n_paths']
    evaluation_frequency = variant['evaluation_frequency']

    policy_params = variant['alg_params']

    min_memory_size = policy_params['min_memory_size']
    steps_per_cycle = policy_params['steps_per_cycle']
    train_per_cycle = policy_params['train_per_cycle']
    batch_size = policy_params['batch_size']

    s_dim = env.observation_space.shape[0] # dimension of state (3 for Battery)

    a_dim = env.action_space.shape[0] # action space dimension (1 or 2)
    a_upperbound = env.action_space.high
    a_lowerbound = env.action_space.low

    agent = CAC(a_dim,s_dim, policy_params, max_global_steps = max_global_steps)
    # policy.restore(variant['log_path'] + "/0/policy")

    pool_params = {
        's_dim': s_dim,
        'a_dim': a_dim,
        'd_dim': 1,
        'store_last_n_paths': store_last_n_paths,
        'memory_capacity': policy_params['memory_capacity'],
        'min_memory_size': policy_params['min_memory_size'],
        'history_horizon': policy_params['history_horizon'],
        'finite_horizon': policy_params['finite_horizon']
    }
    if 'value_horizon' in policy_params.keys():
        pool_params.update({'value_horizon': policy_params['value_horizon']})
    else:
        pool_params['value_horizon'] = None
    pool = Pool(pool_params)

    # For analyse
    Render = env_params['eval_render']

    # Training setting
    t1 = time.time()
    global_step = 0
    last_training_paths = deque(maxlen=store_last_n_paths)
    training_started = False

    log_path = variant['log_path']
    logger.configure(dir=log_path, format_strs=['csv'])
    logger.logkv('tau', policy_params['tau'])

    logger.logkv('alpha3', policy_params['alpha3'])
    logger.logkv('batch_size', policy_params['batch_size'])
    logger.logkv('target_entropy', agent.target_entropy)

    for i in range(max_episodes):
        print("episode # ", i)
        print("global steps ", global_step)

        current_path = {'rewards': [],
                        'distance': [],
                        'a_loss': [],
                        'alpha': [],
                        'lyapunov_error': [],
                        'entropy': [],
                        'beta':[],
                        'action_distance': [],
                        }


        if global_step > max_global_steps:
            break


        s = env.reset()

        # Random start point

        # traj_id = np.random.randint(0, len(data_trajectories))
        traj_id = np.random.randint(0, variant['num_data_trajectories'])
        # traj_id = 0
        traj = data_trajectories[traj_id]
        # print(len(traj))
        start_point = np.random.randint(0, len(traj))
        # start_point = 0
        s = traj[start_point, 1]

        # current state, theta,next w, desired state
        # this is for decision making
        # 16,1,4,16
        s = np.array([s, traj[start_point, 2], traj[start_point, 4]])
        # print(i, s)

        env.state = s
        env.model.state = traj[start_point, -8:]
        
        ep_steps = min(start_point+1+max_ep_steps, len(traj))
        for j in range(start_point+1,ep_steps):
            if Render:
                env.render()
            delta = np.zeros(3)
            # ###### NOSIE ##############

            # noise = np.random.normal(0, 0.01, 0.01)
            # delta[2:]= noise
            # ########IF Noise env##########
            # s= s + delta
            # a = policy.choose_action(s)

            # ###### BIAS ##############

            # noise = s[0:16]*0.01
            # delta[0:16] = noise


            a = agent.act(torch.tensor([s]).float())
            
            action = a_lowerbound + (a.detach().numpy() + 1.) * (a_upperbound - a_lowerbound) / 2
            # action = traj[j-1,16]

            a_upperbound = env.action_space.high
            a_lowerbound = env.action_space.low

            # Run in simulator
            _, r, done, X_ = env.step(action)
            # The new s= current state,next omega, next state
            s_ = np.array([X_[1][0], traj[j, 2], traj[j,4]])
            
            r = modify_reward(r, s, s_, variant['reward_id'])

            if j%100 == 0:
                print("current state: ", s, "true action: ", traj[j, 5], " predicted action: ", action, " and reward : ", r)

            env.state = s_

            # theta_pre=theta
            if training_started:
                global_step += 1
                agent.scheduler_step()

            if j == max_ep_steps - 1+start_point:
                done = True

            terminal = 1. if done else 0.

            if j>start_point+2:
                pool.store(s, a.detach().numpy().flatten(), np.zeros([1]), np.zeros([1]), r, terminal, s_,_s)

            if pool.memory_pointer > min_memory_size and global_step % steps_per_cycle == 0:
                training_started = True

                for _ in range(train_per_cycle):
                    batch = pool.sample(batch_size)
                    alpha_loss, beta_loss, labda_loss, actor_loss, lyapunov_loss = agent.learn(batch)
                    if j % 200 == 0:
                        print("labda = ", agent.labda, " | alpha = ", agent.alpha, 
                            " | l_loss = ", lyapunov_loss , " | entropy = ", agent.log_pis,
                            " | a_loss = ", actor_loss, " | alpha_loss = ", alpha_loss,
                            " | labda_loss = ", labda_loss)
            if training_started:
                current_path['rewards'].append(r)
                current_path['lyapunov_error'].append(lyapunov_loss.detach().numpy())
                current_path['alpha'].append(agent.alpha.detach().numpy())
                current_path['entropy'].append(entropy)
                current_path['a_loss'].append(actor_loss.detach().numpy())
                current_path['beta'].append(agent.beta.detach().numpy())
                # current_path['action_distance'].append(action_distance)

            if training_started and global_step % evaluation_frequency == 0 and global_step > 0:

                logger.logkv("total_timesteps", global_step)

                training_diagnotic = evaluate_training_rollouts(last_training_paths)
                # print(training_diagnotic)
                if training_diagnotic is not None:
                    print("doing training evaluation")
                    eval_diagnotic = training_evaluation(variant, env, agent)
                    [logger.logkv(key, eval_diagnotic[key]) for key in eval_diagnotic.keys()]
                    training_diagnotic.pop('return')
                    [logger.logkv(key, training_diagnotic[key]) for key in training_diagnotic.keys()]

                    string_to_print = ['time_step:', str(global_step), '|']
                    [string_to_print.extend([key, ':', str(eval_diagnotic[key]), '|'])
                     for key in eval_diagnotic.keys()]
                    [string_to_print.extend([key, ':', str(round(training_diagnotic[key], 2)) , '|'])
                     for key in training_diagnotic.keys()]
                    print(''.join(string_to_print))

                logger.dumpkvs()
                if eval_diagnotic['test_return'] / eval_diagnotic['test_average_length'] <= Min_cost:
                    Min_cost = eval_diagnotic['test_return'] / eval_diagnotic['test_average_length']
                    print("New lowest cost:", Min_cost)
                    agent.save_result(log_path)
                else:
                    print("cost did not improve.")
                    print("The best cost is ", Min_cost)
                    print("avg cost was ", eval_diagnotic['test_return']/eval_diagnotic['test_average_length'])
                if training_started and global_step % (10*evaluation_frequency) == 0 and global_step > 0:
                    agent.save_result(log_path)

            # State Update
            _s = s
            s = s_

            # OUTPUT TRAINING INFORMATION AND LEARNING RATE DECAY
            if done:
                if training_started:
                    last_training_paths.appendleft(current_path)
                break
    agent.save_result(log_path)

    print('Running time: ', time.time() - t1)
    return

def eval(variant):
    env_name = variant['env_name']
    data_trajectories=get_data()
    env = get_env_from_name(env_name)
    env_params = variant['env_params']
    max_ep_steps = env_params['max_ep_steps']
    max_global_steps = env_params['max_global_steps']
    policy_params = variant['alg_params']
    s_dim = env.observation_space.shape[0]
    print("observation_space = ", s_dim)
    a_dim = env.action_space.shape[0]
    print("action space = ", a_dim)
    a_upperbound = env.action_space.high
    print("upper bound =", a_upperbound)
    a_lowerbound = env.action_space.low
    print("lower bound = ", a_lowerbound)

    agent = CAC(a_dim,s_dim, policy_params, max_global_steps = max_global_steps)

    log_path = variant['log_path'] + '/eval/' + str(0)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    
    logger.configure(dir=log_path, format_strs=['csv'])
    agent.restore(variant['log_path'] + '/' + str(0)+'/policy')

    # Training setting
    t1 = time.time()
    PLOT_theta_1 = []
    PLOT_ground_theta_1 = []
    mst=[]
    agent_traj=[]
    ground_traj=[]

    for i in tqdm(range(variant['num_data_trajectories'])):
        traj = data_trajectories[i]

        env.reset()
        cost = 0

        s = traj[0, 1]
        PLOT_state = np.array([s])
        s = np.array([s, traj[0, 2], traj[0, 4]])
        print("initial state : ", s)
        print("action here is : ", traj[0,5])
        env.state = s
        env.model.state = traj[0, -8:]

        ep_steps = len(traj)
        for j in range(1,ep_steps):

            if agent_traj == []:
                agent_traj = [s[0]]
            else:
                agent_traj = np.concatenate((agent_traj, [s[0]]),axis=0)

            if ground_traj == []:
                ground_traj = [traj[j-1,1]]
            else:
                ground_traj = np.concatenate((ground_traj, [traj[j-1,1]]),axis=0)

            delta = np.zeros(3)
            # ###### NOSIE ##############

            # noise = np.random.normal(0, 0.001, 16)
            # delta[20:]= noise

            # ###### BIAS ##############

            # noise = s[0:16]*0.005
            # delta[0:16] = noise

            a = agent.act(torch.tensor([s+delta]).float(), True)
            # print(a)
            action = a_lowerbound + (a.detach().cpu().numpy() + 1.) * (a_upperbound - a_lowerbound) / 2
            # print(action)

            _, r, done, X_ = env.step(action)
            if (j%200 == 0):
                print("X predicted ", X_, " and actual: ", traj[j-1, 4])
                print("predicted action : ", action, ", reward : ", r )

            # print(r)
            # The new s= current state,next omega, next state
            s_ = np.array([X_[1,0], traj[j, 2], traj[j, 4]])

            env.state = s_

            theta = action
            PLOT_theta_1.append(theta[0])
            PLOT_ground_theta_1.append(traj[j, 5])
            mst.append(np.linalg.norm(traj[j, 5] - theta[0]))

            PLOT_state = np.vstack((PLOT_state, np.array([X_[1,0]])))

            logger.logkv('rewards', r)
            logger.logkv('timestep', j)
            logger.dumpkvs()

            cost=cost+r
            if j == len(traj)-1:
                done = True

            s = s_

            if done:
                #print('episode:', i,'trajectory_number:',traj_num,'total_cost:',cost,'steps:',j-start_point)
                break
    x = np.linspace(0,np.shape(PLOT_ground_theta_1)[0]-1,np.shape(PLOT_ground_theta_1)[0])
    # plt.plot(x, PLOT_theta_1, color='blue', label='Tracking')
    # plt.plot(x, PLOT_ground_theta_1, color='black', linestyle='--', label='Ground truth')
    # plt.show()

    fig = plt.figure()
    with h5py.File(variant['log_path'] + '/' +'CAC_theta.h5', 'w') as hdf:
         hdf.create_dataset('Data', data=PLOT_theta_1)
    with h5py.File(variant['log_path'] + '/' +'Normal_theta_ground.h5', 'w') as hdf:
         hdf.create_dataset('Data', data=PLOT_ground_theta_1)
    with h5py.File(variant['log_path'] + '/' +'CAC_track.h5', 'w') as hdf:
         hdf.create_dataset('Data', data=agent_traj)
    with h5py.File(variant['log_path'] + '/' +'GT_track.h5', 'w') as hdf:
         hdf.create_dataset('Data', data=ground_traj)

    plt.plot(x, PLOT_theta_1, color='blue', label='Tracking')
    plt.plot(x, PLOT_ground_theta_1, color='black',linestyle='--',label='Ground truth')
    plt.show()

    plt.plot(x, agent_traj, color='blue', label='Tracking')
    plt.plot(x, ground_traj, color='black',linestyle='--',label='Ground truth')
    plt.show()

    return

def modify_reward(r, s = None, s_ = None, id = 1):
    if id == 1:
        r = r
    if id == 2:
        if r > 1e-4:
            r = r*50
        else:
            r = r
    if id == 3:
        if s_[0] < s_[2]:
            r =  r*50
        elif (r > 1e-4):
            r =  r*20
        else:
            r = r
    if id == 4:
        if s_[0] < s_[2] :
            r = np.exp(r)*50
        elif (r > 1e-4):
            r = np.exp(r)*20
        else:
            r = r
    if id == 5:
        if s[0] < s[2]:
            if s_[0] > s_[2]:
                r = r
            else:
                r = np.exp(r)*20
        else:
            if s_[0] < s_[2]:
                r = np.exp(r)*50
            elif (r > 1e-4):
                r = np.exp(r)*20
            else:
                r = r
    if id == 6:
        if s[0] < s[2]:
            if s_[0] < s_[2]:
                r = r*20
        else:
            if s_[0] < s_[2]:
                r = r*50
            elif (r > 1e-4):
                r = r*20
    if id == 7:
        if r > 1e-4:
            r = r*20
        else:
            r = -r*20

    return r

