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


def get_traj():
    #process = 'DS02_W_X_m_theta_2D_N_tw_1_stride_100_R_early_100'
    #process = 'DS229_W_X_m_theta_2D_N_tw_1_stride_10_R_early_100'
    process = 'DS229_W_X_m_theta_2D_N_tw_1_stride_10_R_early_100'


    # with h5py.File('data/' + 'CMAPPS_Prognostics_' + process + '_Train.h5', 'r') as hdf:
    with h5py.File('data/' + 'CMAPPS_Surrogate_' + process+'.h5', 'r') as hdf:
        data = hdf.get('X_') # dim = 16
        X_ = np.array(data)
        data = hdf.get('X')  # 16
        X = np.array(data)
        data = hdf.get('T')  # 10
        T = np.array(data)
        data = hdf.get('W')  # 4
        W = np.array(data)
    data_stack = np.concatenate((X, T, W, X_), axis=-1)
    # print(np.shape(data_stack))
    return data_stack



def training_evaluation(variant, env, policy):

    env_name = variant['env_name']
    traj = get_traj()
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
    traj_num = np.random.randint(0, 1000000)
    print("Test on trajectory:", 2)
    for i in range(variant['store_last_n_paths']):

        cost = 0
        s = env.reset()

        # Random start point
        start_point = np.random.randint(0, 500000)
        # start_point = 222222  # The ground truth on the test traj is 15.03391147682846/new 14.4
        # start_point = 333333  # The ground truth on the test traj is 15.03391147682846

        s = traj[start_point, :16]

        # current state, next omega, desired state
        # this is for decision making

        s = np.concatenate([[s], [traj[start_point, 17:]]], axis=1)[0]


        env.state = s


        for j in range(start_point+1,max_ep_steps+start_point+1):

            if Render:
                env.render()
            # start = time.time()
            a = policy.choose_action(s, True)
            # end = time.time()
            # print(end-start)

            action = a_lowerbound + (a + 1.) * (a_upperbound - a_lowerbound) / 2
            # action = traj[j-1,16]

            X_, r, done, theta = env.step(action)
            # The new s= current state,next omega, next state
            s_ = np.concatenate([X_, [traj[j, 17:]]], axis=1)[0]
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

