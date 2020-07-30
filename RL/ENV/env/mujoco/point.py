import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

import math

class PointEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'point.xml', 5)
        utils.EzPickle.__init__(self)
        self.max_reward = 1500.
    def step(self, action):
        size=40
        # variables = [i for i in dir(self.sim.data)]
        # print(self.sim.get_state())
        # pos_before = np.copy(self.get_body_com("torso"))
        # self.do_simulation(action, self.frame_skip)
        # pos_after = np.copy(self.get_body_com("torso"))
        #action ballx and rot
        #qpos

        sim_state = self.sim.get_state()
        pos_before=np.copy(self.get_body_com("torso"))
        x_joint_i = self.sim.model.get_joint_qpos_addr("ballx")    #x位移
        y_joint_i = self.sim.model.get_joint_qpos_addr("bally")    #x位移
        z_joint_i = self.sim.model.get_joint_qpos_addr("rot")      #角度

        # 角度
        sim_state.qpos[z_joint_i] += action[1]
        ori = sim_state.qpos[z_joint_i]

        # compute increment in each direction
        dx = math.cos(ori) * action[0]
        dy = math.sin(ori) * action[0]
        # ensure that the robot is within reasonable range
        sim_state.qpos[x_joint_i] =  np.clip(sim_state.qpos[x_joint_i] + dx, -size, size)
        sim_state.qpos[y_joint_i] =  np.clip(sim_state.qpos[y_joint_i] + dy, -size, size)
        # ensure that the robot is within reasonable range
        self.sim.set_state(sim_state)
        self.sim.forward()
        next_obs = self.get_current_obs()
        pos_after = np.copy(self.get_body_com("torso"))
        self.circle_mode=True
        self.target_dist=15

        if self.circle_mode:
            # vel = (pos_after-pos_before)/self.model.opt.timestep
            x, y = pos_after[0], pos_after[1]
            # dx, dy = vel[0], vel[1]
            reward = -y * dx + x * dy
            reward /= (1 + np.abs(np.sqrt(x ** 2 + y ** 2) - self.target_dist))

        next_obs = self.get_current_obs()
        done = False
        if abs(x) >3 :
            violation_of_constraint = 1
        else:
            violation_of_constraint = 0
        l_reward = max(abs(x)- 0.8*3., 0.)
        # if abs(x) >0.8*3:
        #     l_reward = 1.
        # else:
        #     l_reward = 0.
        return next_obs, reward, done, dict(l_rewards=l_reward,violation_of_constraint= violation_of_constraint)

    def get_current_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flatten(),
            self.sim.data.qvel.flat,
            self.get_body_com("torso").flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        return self.get_current_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
# from rllab.envs.base import Step
# from .mujoco_env_rllab import MujocoEnv
# from rllab.core.serializable import Serializable
# from rllab.misc.overrides import overrides
# import numpy as np
# import math
# from rllab.mujoco_py import glfw
#
#
# from rllab.envs.base import Step
# from rllab.envs.mujoco.mujoco_env import MujocoEnv
# from rllab.core.serializable import Serializable
# from rllab.misc.overrides import overrides
# import numpy as np
# import math
# from rllab.mujoco_py import glfw
#
# class PointEnv(MujocoEnv, Serializable):
#
#     """
#     Use Left, Right, Up, Down, A (steer left), D (steer right)
#     """
#
#     FILE = 'point.xml'
#
#     def __init__(self,
#             size=40,
#             align_mode=True,
#             reward_dir=[0.,0.],
#             target_dist=5.,
#             *args, **kwargs):
#         self.size = size
#         self.align_mode = align_mode
#         self.reward_dir = reward_dir
#         self.target_dist = target_dist
#         super(PointEnv, self).__init__(*args, **kwargs)
#         Serializable.quick_init(self, locals())
#
#     def get_current_obs(self):
#         return np.concatenate([
#             self.model.data.qpos.flatten(),
#             self.model.data.qvel.flat,
#             self.get_body_com("torso").flat,
#         ])
#
#     def step(self, action):
#         qpos = np.copy(self.model.data.qpos)
#         qpos[2, 0] += action[1]
#         ori = qpos[2, 0]
#         # compute increment in each direction
#         dx = math.cos(ori) * action[0]
#         dy = math.sin(ori) * action[0]
#         # ensure that the robot is within reasonable range
#         qpos[0, 0] = np.clip(qpos[0, 0] + dx, -self.size, self.size)
#         qpos[1, 0] = np.clip(qpos[1, 0] + dy, -self.size, self.size)
#         self.model.data.qpos = qpos
#         self.model.forward()
#         next_obs = self.get_current_obs()
#         self.circle_mode=True
#         self.target_dist=15
#         if self._circle_mode:
#             pos = self.wrapped_env.get_body_com("torso")
#             vel = self.wrapped_env.get_body_comvel("torso")
#             dt = self.wrapped_env.model.opt.timestep
#             x, y = pos[0], pos[1]
#             dx, dy = vel[0], vel[1]
#             reward = -y * dx + x * dy
#             reward /= (1 + np.abs(np.sqrt(x ** 2 + y ** 2) - self._target_dist))
#         return next_obs, reward, done, dict(l_rewards=0)
#
#     def get_xy(self):
#         qpos = self.model.data.qpos
#         return qpos[0, 0], qpos[1, 0]
#
#     def set_xy(self, xy):
#         qpos = np.copy(self.model.data.qpos)
#         qpos[0, 0] = xy[0]
#         qpos[1, 0] = xy[1]
#         self.model.data.qpos = qpos
#         self.model.forward()
#
#     @overrides
#     def action_from_key(self, key):
#         lb, ub = self.action_bounds
#         if key == glfw.KEY_LEFT:
#             return np.array([0, ub[0]*0.3])
#         elif key == glfw.KEY_RIGHT:
#             return np.array([0, lb[0]*0.3])
#         elif key == glfw.KEY_UP:
#             return np.array([ub[1], 0])
#         elif key == glfw.KEY_DOWN:
#             return np.array([lb[1], 0])
#         else:
#             return np.array([0, 0])
#

# from sklearn.metrics import mean_squared_error
import numpy as np
# import pandas as pd
import tensorflow as tf
from tqdm import tqdm


# Import data
data = np.genfromtxt("T0/data/train.csv",dtype=float, delimiter=',',skip_header=1)[:,1:]

# Split train&test

train_data = data[0:9000,:]

test_data = data[9000:,:]



# Training setting
training_epochs = 1000
batch_size = 1024
learning_rate = 0.1

# Model
class DNN(object):
    def __init__(self, x_dim,y_dim):
        tf.reset_default_graph()
        ###############################  Model parameters  ####################################

        self.x_dim = x_dim
        self.y_dim = y_dim
        self.hidden_dim = 1

        self.x = tf.placeholder(tf.float32, [None, self.x_dim], 'x')
        # self.y_predict = tf.placeholder(tf.float32, [None, self.y_dim], 'y')
        self.y_real = tf.placeholder(tf.float32, [None, self.y_dim], 'y_real')
        self.y_predict = self.dnn(self.x)
        self.LR = tf.placeholder(tf.float32, None, 'LR')

        # prediction loss (objective function)
        self.MSE = tf.reduce_mean(tf.losses.mean_squared_error(self.y_predict,self.y_real))
        self.optimizer = tf.train.AdamOptimizer(self.LR).minimize(self.MSE)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.opt = [self.optimizer]

    def learn(self, batch_xs,batch_ys,lr):
        _, MSE = self.sess.run([self.optimizer, self.MSE], feed_dict={self.x: batch_xs,self.y_real:batch_ys,self.LR:lr})
        return MSE

    def regression(self,batch_xs,batch_ys):
        pred=self.sess.run([self.y_predict,self.MSE], feed_dict={self.x: batch_xs,self.y_real:batch_ys})
        return pred


    # Dynamic model
# Dynamic model
    def dnn(self, x,reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Dream_NN', reuse=reuse, custom_getter=custom_getter):
            w1_s = tf.get_variable('w1_s', [self.x_dim, self.hidden_dim], trainable=trainable)
            b1 = tf.get_variable('b1', [1, self.hidden_dim], trainable=trainable)
            net_0 = tf.matmul(x, w1_s) + b1
            # net_0 = tf.nn.leaky_relu(tf.matmul(x, w1_s)+b1)  # non linear trans, activation function
            # net_1 = tf.layers.dense(net_0, self.hidden_dim, activation=tf.nn.leaky_relu, name='l2', trainable=trainable)
            # y = tf.layers.dense(net_1, self.y_dim, activation=tf.nn.leaky_relu, name='l3', trainable=trainable)
            return net_0

    def save_result(self, path):
        save_path = self.saver.save(self.sess, path + "/model.ckpt")
        print("Save to path: ", save_path)
    def load_result(self):
        self.saver.restore(self.sess, "Dream/model.ckpt")


# Init model
model=DNN(10,1)

# Training
for epoch in tqdm(range(training_epochs)):
    i=0
    np.random.shuffle(train_data)
    while i < np.shape(train_data)[0]:
        batch_xs = train_data[i:i + batch_size,1:]
        batch_ys = train_data[i:i + batch_size,0:1]
        MSE = model.learn(batch_xs, batch_ys,learning_rate)
        i = i + batch_size
    learning_rate = max(learning_rate * 0.999, 0.0001)

print("Training error:", MSE)


# Test
_,test_MSE=model.regression(test_data[:,1:],test_data[:,0:1])
print("Test error:",test_MSE)
# RMSE = mean_squared_error(y, y_pred)**0.5