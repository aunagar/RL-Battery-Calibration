"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import csv

import tensorflow as tf
import scipy.io as sio
import numpy as np
import tensorflow_probability as tfp

import matplotlib.pyplot as plt
from sklearn import preprocessing
import os
import h5py

# Dynamic model
def dream_nn():
    from keras.models import model_from_json
    from keras import backend as K

    K.clear_session()
    # Supervised
    with open("dnn_models/model_FF_0.json", "r") as json_file:
        loaded_model_json = json_file.read()
    supervised = model_from_json(loaded_model_json)
    supervised.load_weights('dnn_models/model_FF_0.h5')
    print('')
    print("Loaded supervised model from disk")

    return supervised

class CMAPSS(gym.Env):
    def __init__(self):
        # Action after normalized
        # 1 dimension



        # State after normalized
        # X 8  omega 3 X_ 8
        # 8 dimensions



        self.high = np.ones(36)
        self.low = -np.ones(36)

        self.is_discrete_action = False
        if self.is_discrete_action:
            self.actions = [1]

        else:
            self.action_space = spaces.Box(low=np.array([-1.]), high=np.array([1.]), dtype=np.float32)
        self.observation_space = spaces.Box(-self.high, self.high, dtype=np.float32)
        self.model=dream_nn()

        self.seed()
        self.viewer = None
        self.state = None
        self.for_plot=[]
        self.steps_beyond_done = None


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def step(self, action,theta_pre=None):

        if self.is_discrete_action:
            action = self.actions[action]
        else:
            action = np.clip(action, self.action_space.low, self.action_space.high)


        theta=action
        #10

        # state= X,W,X_
        theta_in=theta
        # print(theta_pre,theta_in)

        input=np.concatenate([[self.state[16:20]], [self.state[0:16]]], axis=1)
        input = np.concatenate([input, [theta_in]], axis=1)

        #X_=Dyn(X,t,w)
        #w,x,t


        # Environment noisy
        X_pred=self.model.predict(input)
        #X_pred= X_pred+np.random.normal(0,0.1, 16)*X_pred
        X_pred = X_pred + 0.02* X_pred
        done = False


        # We hope the state is near with desired state and theta is same with last theta
        # 16 4 16

        cost_1 = np.linalg.norm(self.state[20:] - X_pred)
        # print(cost_1)

        # Robust version
        # if cost_1<=0.05:
        #     cost_1=cost_1
        # else:
        #     cost_1=5*cost_1

        #
        # #Basic version
        #
        # if cost_1<=0.01:
        #     cost_1=cost_1
        # elif cost_1>0.01 and cost_1<=0.05:
        #     cost_1=2.5*cost_1
        # elif cost_1>0.05 and cost_1<=0.1:
        #     cost_1=5*cost_1
        # else:
        #     cost_1=10*cost_1


        # #New version
        #
        if cost_1<=0.01:
            cost_1=cost_1
        else:
            cost_1=10*cost_1


        cost = cost_1

        return X_pred, cost, done,theta
    def reset(self):
        self.state = self.np_random.uniform(low=-self.high/2, high=self.high/2, size=(36,))
        self.steps_beyond_done = None
        return np.array(self.state)

