import tensorflow as tf
import os
from variant import *

import numpy as np
import time
import logger
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn import preprocessing
import h5py
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from variant import VARIANT

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
    val_frac = 0.

    data_stack = [np.concatenate((X[i], W[i], X_[i], T[i], state[i]), axis = -1) for i in range(int(val_frac*len(X)), len(X))]
    return data_stack

def training_evaluation(variant, env, policy):

    env_name = variant['env_name']
    data_trajectories = get_data()
    env_params = variant['env_params']

    max_ep_steps = env_params['max_ep_steps']

    a_upperbound = env.action_space.high
    a_lowerbound = env.action_space.low
    # For analyse
    Render = env_params['eval_render']

    # Training setting

    total_cost = []


    episode_length = []

    die_count = 0
    seed_average_cost = []
    print("Test on trajectory:", 2)
    for i in range(variant['store_last_n_paths']):

        cost = 0
        s = env.reset()

        # Random start point
        # traj_id = np.random.randint(0, len(data_trajectories))
        traj_id = 10*i
        traj = data_trajectories[traj_id]

        # start_point = np.random.randint(0, len(traj))
        start_point = 0
        s = traj[start_point, 1]

        # current state, next omega, desired state
        # this is for decision making

        s = np.array([s, traj[start_point, 2], traj[start_point, 4]])
        env.state = s
        env.model.state = traj[start_point, -8:]
        ep_steps = min(len(traj), 3200)
        # ep_steps = min(start_point+max_ep_steps+1, len(traj))
        for j in range(start_point+1, ep_steps):

            if Render:
                env.render()
            # start = time.time()
            a = policy.choose_action(s, True)
            # end = time.time()
            # print(end-start)

            action = a_lowerbound + (a + 1.) * (a_upperbound - a_lowerbound) / 2
            # action = traj[j-1,16]

            _, r, done, X_ = env.step(action)
            # The new s= current state,next omega, next state
            s_ = np.array([X_[1][0], traj[j, 2], traj[j,4]])
            # s_ = np.concatenate([[s_], [theta]], axis=1)[0]
            # s_ = np.concatenate([X_,[[theta]], [traj[j, 9:]]], axis=1)[0]
            env.state = s_
            # theta_pre = theta

            cost += r


            if j == max_ep_steps+start_point - 1:
                done = True
            s = s_


            if done:
                seed_average_cost.append(cost)
                episode_length.append(j-start_point)
                if j < max_ep_steps-1:
                    die_count += 1
                break

    total_cost.append(np.mean(seed_average_cost))

    total_cost_std = np.std(total_cost, axis=0)
    total_cost_mean = np.average(total_cost)

    average_length = np.average(episode_length)

    diagnostic = {'test_return': total_cost_mean,
                  'test_average_length': average_length}
    return diagnostic

