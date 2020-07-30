import tensorflow as tf
import numpy as np
import time
from .squash_bijector import SquashBijector
from .utils import evaluate_training_rollouts
import tensorflow_probability as tfp
from collections import OrderedDict, deque
import os
from copy import deepcopy
import sys
sys.path.append("..")
from robustness_eval import training_evaluation

from pool.pool import Pool
import logger
from tqdm import tqdm
from variant import *
import scipy.io as sio
from sklearn import preprocessing
import h5py
import matplotlib.pyplot as plt
# SCALE_DIAG_MIN_MAX = (-20,-2.5)
SCALE_DIAG_MIN_MAX = (-2,1)
SCALE_lambda_MIN_MAX = (0, 1)
SCALE_beta_MIN_MAX = (0, 1)


def get_traj():
    process = 'DS229_W_X_m_theta_2D_N_tw_1_stride_10_R_early_100'
    # with h5py.File('data/' + 'CMAPPS_Prognostics_' + process + '_Train.h5', 'r') as hdf:
    with h5py.File('data/' + 'CMAPPS_Surrogate_' + process+'.h5', 'r') as hdf:
        data = hdf.get('X_') # dim = 16
        X_ = np.array(data)
        data = hdf.get('X')  # 16
        X = np.array(data)
        data = hdf.get('T')  # 1
        T = np.array(data)
        data = hdf.get('W')  # 4
        W = np.array(data)
    data_stack = np.concatenate((X, T, W, X_), axis=-1)
    # print(np.shape(data_stack))
    return data_stack

class LAC(object):
    def __init__(self,
                 a_dim,
                 s_dim,
                 variant,
                 action_prior = 'uniform',
                 ):
        ###############################  Model parameters  ####################################
        # self.memory_capacity = variant['memory_capacity']

        self.batch_size = variant['batch_size']
        gamma = variant['gamma']

        tau = variant['tau']
        self.approx_value = True if 'approx_value' not in variant.keys() else variant['approx_value']
        # self.memory = np.zeros((self.memory_capacity, s_dim * 2 + a_dim+ d_dim + 3), dtype=np.float32)
        # self.pointer = 0
        self.sess = tf.Session()
        self._action_prior = action_prior
        s_dim = s_dim * (variant['history_horizon']+1) # if longer lookback
        self.a_dim, self.s_dim, = a_dim, s_dim
        self.history_horizon = variant['history_horizon'] # length of lookback
        self.working_memory = deque(maxlen=variant['history_horizon']+1) # queue to store history
        target_entropy = variant['target_entropy'] # for SAC formulation
        if target_entropy is None:
            self.target_entropy = -self.a_dim  #lower bound of the policy entropy
        else:
            self.target_entropy = target_entropy
        self.target_variance = 0.0
        self.finite_horizon = variant['finite_horizon']
        self.soft_predict_horizon = variant['soft_predict_horizon']
        with tf.variable_scope('Actor'):
            self.S = tf.placeholder(tf.float32, [None, s_dim], 's') # current state
            self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_') # next state
            self._S = tf.placeholder(tf.float32, [None, s_dim], '_s') # prev state
            self.a_input = tf.placeholder(tf.float32, [None, a_dim], 'a_input') # current action
            self.a_input_ = tf.placeholder(tf.float32, [None, a_dim], 'a_input_') # next action
            self._a_input = tf.placeholder(tf.float32, [None, a_dim], '_a_input') # prev action
            self.R = tf.placeholder(tf.float32, [None, 1], 'r') # reward
            self.R_N_ = tf.placeholder(tf.float32, [None, 1], 'r_N_') # Finite Horizon reward
            self.V = tf.placeholder(tf.float32, [None, 1], 'v') # Value function
            self.terminal = tf.placeholder(tf.float32, [None, 1], 'terminal') # is terminal state?
            self.LR_A = tf.placeholder(tf.float32, None, 'LR_A') # Actor learning rate
            self.LR_lag = tf.placeholder(tf.float32, None, 'LR_lag') # Lagrangian update rate
            self.LR_C = tf.placeholder(tf.float32, None, 'LR_C') # Critic Learning rate
            self.LR_L = tf.placeholder(tf.float32, None, 'LR_L') # Lyapunov Learning rate
            # self.labda = tf.placeholder(tf.float32, None, 'Lambda')
            labda = variant['labda']
            alpha = variant['alpha']
            alpha3 = variant['alpha3']

            log_labda = tf.get_variable('lambda', None, tf.float32, initializer=tf.log(labda))
            log_alpha = tf.get_variable('alpha', None, tf.float32, initializer=tf.log(alpha))  # Entropy Temperature

            # The update is in log space
            self.labda = tf.clip_by_value(tf.exp(log_labda), *SCALE_lambda_MIN_MAX)
            self.alpha = tf.exp(log_alpha)

            self.a, self.deterministic_a, self.a_dist,self.ori_dis = self._build_a(self.S, )  # 这个网络用于及时更新参数

            self.l = self._build_l(self.S, self.a_input)   # lyapunov 网络


            self.use_lyapunov = variant['use_lyapunov']
            self.adaptive_alpha = variant['adaptive_alpha']
            

            a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor/actor')
            l_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor/Lyapunov')

            ###############################  Model Learning Setting  ####################################
            ema = tf.train.ExponentialMovingAverage(decay=1 - tau)  # soft replacement

            def ema_getter(getter, name, *args, **kwargs):
                return ema.average(getter(name, *args, **kwargs))
            target_update = [ema.apply(a_params),  ema.apply(l_params)]  # soft update operation

            # This network does not update parameters
            # Used for prediction Critic of Q_target Neutral action
            a_, _, a_dist_,_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)  # replaced target parameters

            lya_a_, _, lya_a_dist_,_ = self._build_a(self.S_, reuse=True)
            _, d_a, _, near_dis= self._build_a(self._S, reuse=True)
            # self.cons_a_input_ = tf.placeholder(tf.float32, [None, a_dim, 'cons_a_input_'])
            # self.log_pis = log_pis = self.a_dist.log_prob(self.a)
            self.log_pis = log_pis = self.a_dist.log_prob(self.a)
            self.prob = tf.reduce_mean(self.a_dist.prob(self.a))

            # 这个网络不及时更新参数, 用于给出 Actor 更新参数时的 Gradient ascent 强度

            l_ = self._build_l(self.S_, a_, reuse=True, custom_getter=ema_getter)
            self.l_ = self._build_l(self.S_, lya_a_, reuse=True)

            # Output info

            self.kl_loss = tf.reduce_mean(tf.minimum(self.ori_dis.kl_divergence(near_dis), tf.constant(1000.)))
            self.distance= tf.reduce_mean(tf.norm(self.S - self._S))


            # Loss function
            self.l_derta = tf.reduce_mean(self.l_ - self.l + (alpha3) * self.R)
            labda_loss = -tf.reduce_mean(log_labda * self.l_derta)
            self.l_action = tf.reduce_mean(tf.norm(d_a - self.deterministic_a))
            alpha_loss = -tf.reduce_mean(log_alpha * tf.stop_gradient(log_pis + self.target_entropy))



            self.alpha_train = tf.train.AdamOptimizer(self.LR_A
                                                      ).minimize(alpha_loss, var_list=log_alpha)
            self.lambda_train = tf.train.AdamOptimizer(self.LR_lag).minimize(labda_loss, var_list=log_labda)



            if self._action_prior == 'normal':
                policy_prior = tf.contrib.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(self.a_dim),
                    scale_diag=tf.ones(self.a_dim))
                policy_prior_log_probs = policy_prior.log_prob(self.a)
            elif self._action_prior == 'uniform':
                policy_prior_log_probs = 0.0

            if self.use_lyapunov is True:
                # The l_derta, the smaller the better
                a_loss = self.labda * self.l_derta + self.alpha * tf.reduce_mean(log_pis) - policy_prior_log_probs
            else:
                a_loss = a_preloss

            self.a_loss = a_loss
            self.atrain = tf.train.AdamOptimizer(self.LR_A).minimize(a_loss, var_list=a_params)

            next_log_pis = a_dist_.log_prob(a_)
            with tf.control_dependencies(target_update):  # soft replacement happened at here
                if self.approx_value:
                    if self.finite_horizon:
                        if self.soft_predict_horizon:
                            l_target = self.R - self.R_N_ + tf.stop_gradient(l_)
                        else:
                            l_target = self.V
                    else:
                        l_target = self.R + gamma * (1-self.terminal)*tf.stop_gradient(l_)  # Lyapunov critic - self.alpha * next_log_pis
                else:
                    l_target = self.R

                self.l_error = tf.losses.mean_squared_error(labels=l_target, predictions=self.l)
                self.ltrain = tf.train.AdamOptimizer(self.LR_L).minimize(self.l_error, var_list=l_params)

            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()
            self.diagnotics = [self.labda, self.alpha, self.l_error, tf.reduce_mean(-self.log_pis), self.a_loss,self.l_action,self.kl_loss,self.distance]

            if self.use_lyapunov is True:
                self.opt = [self.ltrain, self.lambda_train]
            self.opt.append(self.atrain)
            if self.adaptive_alpha is True:
                self.opt.append(self.alpha_train)

    def choose_action(self, s, evaluation = False):
        if len(self.working_memory) < self.history_horizon:
            [self.working_memory.appendleft(s) for _ in range(self.history_horizon)]

        self.working_memory.appendleft(s)
        try:
            s = np.concatenate(self.working_memory)
        except ValueError:
            print(s)

        if evaluation is True:
            try:
                return self.sess.run(self.deterministic_a, {self.S: s[np.newaxis, :]})[0]
            except ValueError:
                return
        else:
            return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self, LR_A, LR_C, LR_L,LR_lag, batch):

        bs = batch['s']  # state
        ba = batch['a']  # action

        br = batch['r']  # reward

        bterminal = batch['terminal']
        bs_ = batch['s_']  # next state
        b_s = batch['_s']
        feed_dict = {self.a_input: ba,  self.S: bs, self.S_: bs_, self.R: br, self.terminal: bterminal,
                     self.LR_C: LR_C, self.LR_A: LR_A, self.LR_L: LR_L, self.LR_lag:LR_lag,self._S:b_s}
        if self.finite_horizon:
            bv = batch['value']
            b_r_ = batch['r_N_']
            feed_dict.update({self.V:bv, self.R_N_:b_r_})

        self.sess.run(self.opt, feed_dict)
        labda, alpha, l_error, entropy, a_loss,variance,kl,distance = self.sess.run(self.diagnotics, feed_dict)

        return labda, alpha, l_error, entropy, a_loss,variance,kl,distance

    def store_transition(self, s, a,d, r, l_r, terminal, s_,_s):
        transition = np.hstack((s, a, d, [r], [l_r], [terminal], s_,_s))
        index = self.pointer % self.memory_capacity  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, name='actor', reuse=None, custom_getter=None):
        if reuse is None:
            trainable = True
        else:
            trainable = False

        with tf.variable_scope(name, reuse=reuse, custom_getter=custom_getter):
            batch_size = tf.shape(s)[0]
            squash_bijector = (SquashBijector())
            base_distribution = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(self.a_dim), scale_diag=tf.ones(self.a_dim))
            epsilon = base_distribution.sample(batch_size)
            ## Construct the feedforward action
            net_0 = tf.layers.dense(s, 256, activation=tf.nn.leaky_relu, name='l1', trainable=trainable)  # 原始是30
            net_1 = tf.layers.dense(net_0, 256, activation=tf.nn.leaky_relu, name='l2', trainable=trainable)  # 原始是30
            net_2 = tf.layers.dense(net_1, 256, activation=tf.nn.leaky_relu, name='l3', trainable=trainable)  # 原始是30
            # net_3 = tf.layers.dense(net_2, 128, activation=tf.nn.leaky_relu, name='l4', trainable=trainable)  # 原始是30
            # net_0 = tf.layers.dense(s, 64, activation=tf.nn.leaky_relu, name='l1', trainable=trainable)  # 原始是30
            #
            # net_2 = tf.layers.dense(net_0, 64, activation=tf.nn.leaky_relu, name='l4', trainable=trainable)  # 原始是30
            mu = tf.layers.dense(net_2, self.a_dim, activation= None, name='a', trainable=trainable)
            log_sigma = tf.layers.dense(net_2, self.a_dim, None, trainable=trainable)
            log_sigma = tf.clip_by_value(log_sigma, *SCALE_DIAG_MIN_MAX)
            sigma = tf.exp(log_sigma)


            bijector = tfp.bijectors.Affine(shift=mu, scale_diag=sigma) #\pi_theta = mu + sigma*epsilon
            raw_action = bijector.forward(epsilon)
            clipped_a = squash_bijector.forward(raw_action) #tanh(raw_action)

            ## Construct the distribution
            bijector = tfp.bijectors.Chain((
                squash_bijector,
                tfp.bijectors.Affine(
                    shift=mu,
                    scale_diag=sigma),
            ))
            distribution = tfp.distributions.ConditionalTransformedDistribution(
                    distribution=base_distribution,
                    bijector=bijector)
            ori_dis = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=sigma) #original distribution

            clipped_mu = squash_bijector.forward(mu)

        return clipped_a, clipped_mu, distribution,ori_dis

    def _build_adv(self, s, name='Adv', reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope(name, reuse=reuse, custom_getter=custom_getter):
            net_0 = tf.layers.dense(s, 256, activation=tf.nn.relu, name='l1', trainable=trainable)  # 原始是30
            net_1 = tf.layers.dense(net_0, 256, activation=tf.nn.relu, name='l2', trainable=trainable)  # 原始是30
            return tf.layers.dense(net_1, self.s_dim, activation=tf.nn.tanh,trainable=trainable)  # Q(s,a)

    def evaluate_value(self, s, a):

        if len(self.working_memory) < self.history_horizon:
            [self.working_memory.appendleft(s) for _ in range(self.history_horizon)]

        self.working_memory.appendleft(s)
        try:
            s = np.concatenate(self.working_memory)
        except ValueError:
            print(s)

        return self.sess.run(self.l, {self.S: s[np.newaxis, :], self.a_input: a[np.newaxis, :]})[0]

    def _build_l(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Lyapunov', reuse=reuse, custom_getter=custom_getter):
            n_l1 = 256#30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net_0 = tf.nn.leaky_relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net_1 = tf.layers.dense(net_0, 256, activation=tf.nn.leaky_relu, name='l2', trainable=trainable)  # Original 30
            net_2 = tf.layers.dense(net_1, 256, activation=tf.nn.leaky_relu, name='l3', trainable=trainable)  # Original 30

            return tf.expand_dims(tf.reduce_sum(tf.square(net_2), axis=1),axis=1)  # Q(s,a)

    def save_result(self, path):

        save_path = self.saver.save(self.sess, path + "/policy/model.ckpt")
        print("Save to path: ", save_path)

    def restore(self, path):
        model_file = tf.train.latest_checkpoint(path+'/')
        if model_file is None:
            success_load = False
            return success_load
        self.saver.restore(self.sess, model_file)
        success_load = True
        print("Load successful, model file:", model_file)
        print("#########################################################")
        return success_load

def train(variant):
    Min_cost = 1000000

    traj=get_traj()
    env_name = variant['env_name']
    env = get_env_from_name(env_name)

    env_params = variant['env_params']

    max_episodes = env_params['max_episodes']
    max_ep_steps = env_params['max_ep_steps']
    max_global_steps = env_params['max_global_steps']
    store_last_n_paths = variant['store_last_n_paths']
    evaluation_frequency = variant['evaluation_frequency']

    policy_params = variant['alg_params']

    min_memory_size = policy_params['min_memory_size']
    steps_per_cycle = policy_params['steps_per_cycle']
    train_per_cycle = policy_params['train_per_cycle']
    batch_size = policy_params['batch_size']

    lr_a, lr_c, lr_l = policy_params['lr_a'], policy_params['lr_c'], policy_params['lr_l']
    lr_a_now = lr_a  # learning rate for actor
    lr_c_now = lr_c  # learning rate for critic
    lr_l_now = lr_l  # learning rate for critic


    s_dim = env.observation_space.shape[0]


    a_dim = env.action_space.shape[0]
    a_upperbound = env.action_space.high
    a_lowerbound = env.action_space.low


    policy = LAC(a_dim,s_dim, policy_params)



    pool_params = {
        's_dim': s_dim,
        'a_dim': a_dim,
        'd_dim': 1,
        'store_last_n_paths': store_last_n_paths,
        'memory_capacity': policy_params['memory_capacity'],
        'min_memory_size': policy_params['min_memory_size'],
        'history_horizon': policy_params['history_horizon'],
        'finite_horizon':policy_params['finite_horizon']
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
    logger.logkv('target_entropy', policy.target_entropy)

    for i in range(max_episodes):



        current_path = {'rewards': [],
                        'distance': [],
                        'kl_divergence': [],
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

        start_point = np.random.randint(0, 500000)

        s = traj[start_point, :16]

        # current state, next omega, desired state
        # this is for decision making
        # 16,1,4,16
        s = np.concatenate([[s], [traj[start_point, 17:]]], axis=1)[0]

        env.state = s


        for j in range(start_point+1,start_point+1+max_ep_steps):
            if Render:
                env.render()

            a = policy.choose_action(s)

            action = a_lowerbound + (a + 1.) * (a_upperbound - a_lowerbound) / 2

            a_upperbound = env.action_space.high
            a_lowerbound = env.action_space.low

            # Run in simulator
            X_, r, done, theta = env.step(action)
            # The new s= current state,next omega, next state
            s_ = np.concatenate([X_, [traj[j, 17:]]], axis=1)[0]
            # s_ = np.concatenate([[s_], [theta]], axis=1)[0]
            # s_ = np.concatenate([X_,[[theta]], [traj[j, 9:]]], axis=1)[0]
            env.state = s_

            # theta_pre=theta

            if training_started:
                global_step += 1

            if j == max_ep_steps - 1+start_point:
                done = True

            terminal = 1. if done else 0.

            if j>start_point+2:
                pool.store(s, a, np.zeros([1]), np.zeros([1]), r, terminal, s_,_s)
            # policy.store_transition(s, a, disturbance, r,0, terminal, s_)

            if pool.memory_pointer > min_memory_size and global_step % steps_per_cycle == 0:
                training_started = True

                for _ in range(train_per_cycle):
                    batch = pool.sample(batch_size)
                    labda, alpha, l_loss, entropy, a_loss,action_distance,kl,distance = policy.learn(lr_a_now, lr_c_now, lr_l_now, lr_a_now/10, batch)

            if training_started:
                current_path['rewards'].append(r)
                current_path['distance'].append(distance)
                current_path['kl_divergence'].append(kl)
                current_path['lyapunov_error'].append(l_loss)
                current_path['alpha'].append(alpha)
                current_path['entropy'].append(entropy)
                current_path['a_loss'].append(a_loss)
                current_path['beta'].append(1)
                current_path['action_distance'].append(action_distance)






            if training_started and global_step % evaluation_frequency == 0 and global_step > 0:

                logger.logkv("total_timesteps", global_step)

                training_diagnotic = evaluate_training_rollouts(last_training_paths)
                # print(training_diagnotic)
                if training_diagnotic is not None:
                    eval_diagnotic = training_evaluation(variant, env, policy)
                    [logger.logkv(key, eval_diagnotic[key]) for key in eval_diagnotic.keys()]
                    training_diagnotic.pop('return')
                    [logger.logkv(key, training_diagnotic[key]) for key in training_diagnotic.keys()]
                    logger.logkv('lr_a', lr_a_now)
                    logger.logkv('lr_c', lr_c_now)
                    logger.logkv('lr_l', lr_l_now)

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
                    policy.save_result(log_path)



            # 状态更新
            _s=s
            s = s_



            # OUTPUT TRAINING INFORMATION AND LEARNING RATE DECAY
            if done:
                if training_started:
                    last_training_paths.appendleft(current_path)


                frac = 1.0 - (global_step - 1.0) / max_global_steps
                lr_a_now = lr_a * frac  # learning rate for actor
                lr_c_now = lr_c * frac  # learning rate for critic
                lr_l_now = lr_l * frac  # learning rate for critic



                break
    policy.save_result(log_path)

    print('Running time: ', time.time() - t1)
    return


def eval(variant):
    env_name = variant['env_name']
    traj=get_traj()
    env = get_env_from_name(env_name)
    env_params = variant['env_params']
    max_ep_steps = env_params['max_ep_steps']
    policy_params = variant['alg_params']
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_upperbound = env.action_space.high
    a_lowerbound = env.action_space.low
    policy = LAC(a_dim, s_dim, policy_params)

    log_path = variant['log_path'] + '/eval/' + str(0)
    logger.configure(dir=log_path, format_strs=['csv'])
    policy.restore(variant['log_path'] + '/' + str(0)+'/policy')

    # Training setting
    t1 = time.time()
    PLOT_theta_1 = []
    PLOT_ground_theta_1 = []
    mst=[]
    agent_traj=[]
    ground_traj=[]

    for i in tqdm(range(500)):


        s = env.reset()
        cost = 0

        traj_num = 0
        # Random start point

        start_point = 0+1000*i

        s = traj[start_point, :16]
        PLOT_state = s
        s = np.concatenate([[s], [traj[start_point, 17:]]], axis=1)[0]

        env.state = s

        for j in range(start_point+1,start_point+1+1000):

            if agent_traj == []:
                agent_traj = s[0:16]
            else:
                agent_traj = np.vstack((agent_traj, s[0:16]))

            if ground_traj == []:
                ground_traj = traj[j-1,0:16]
            else:
                ground_traj = np.vstack((ground_traj, traj[j-1,0:16]))




            a = policy.choose_action(s,True)
            action = a_lowerbound + (a + 1.) * (a_upperbound - a_lowerbound) / 2

            X_, r, done, theta = env.step(action)
            # The new s= current state,next omega, next state
            s_ = np.concatenate([X_, [traj[j, 17:]]], axis=1)[0]

            env.state = s_

            PLOT_theta_1.append(theta[0])
            PLOT_ground_theta_1.append(traj[j, 16])
            mst.append(np.linalg.norm(traj[j, 16] - theta[0]))




            PLOT_state = np.vstack((PLOT_state, X_))



            logger.logkv('rewards', r)
            logger.logkv('timestep', j)
            logger.dumpkvs()

            cost=cost+r
            if j == 1000- 1+start_point:
                done = True

            s = s_

            if done:
                #print('episode:', i,'trajectory_number:',traj_num,'total_cost:',cost,'steps:',j-start_point)
                break
    x = np.linspace(0,np.shape(PLOT_ground_theta_1)[0]-1,np.shape(PLOT_ground_theta_1)[0])

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

    return