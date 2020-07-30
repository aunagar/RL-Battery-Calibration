import gym
import datetime
import numpy as np
import ENV.env
SEED = None

ENV_PARAMS = {
    'CMAPSS': {
        'max_ep_steps': 512, # Maximum steps per episode
        'max_global_steps': int(5e6),
        # 'max_global_steps' : int(500),
        'max_episodes': int(5e6), # Maximum number of episodes for RL training
        'disturbance dim': 8,
        'eval_render': False, },
    'BatteryCalib' : {
        'max_ep_steps' : int(512),
        'max_global_steps' : int(500000),
        'max_episodes' : int(100000),
        'eval_render': False }
}

ALG_PARAMS = {
    'LAC': {
        'iter_of_actor_train_per_epoch': 50,
        'iter_of_disturber_train_per_epoch': 50,
        'memory_capacity': int(1e6),
        'min_memory_size': 1000,
        'batch_size': 256,
        'labda': 1.,
        'alpha': 2.,
        'alpha3': 1,
        'tau': 5e-3,
        'lr_a': 1e-4,
        'lr_c': 3e-4, # Why two critic learning rate?
        'lr_l': 3e-4,
        'gamma': 0.995,
        # 'gamma': 0.75,
        'steps_per_cycle': 1,
        'train_per_cycle': 1,
        'use_lyapunov': True,
        'adaptive_alpha': True,
        'approx_value': True, # This?
        'value_horizon': 2,
        # 'finite_horizon': True,
        'finite_horizon': False,
        'soft_predict_horizon': False,
        'target_entropy': None,
        'history_horizon': 0,  # 0 is using current state only
    },

    'CAC': {
        'iter_of_actor_train_per_epoch': 50,
        'iter_of_disturber_train_per_epoch': 50,
        'memory_capacity': int(1e6),
        'min_memory_size': 10000,
        'batch_size': 256,
        'labda': 1.,
        'alpha': 2.,
        'beta':0.5,
        'alpha3': 1,
        'tau': 5e-3,
        'lr_a': 1e-4,
        'lr_c': 3e-4,
        'lr_l': 3e-4,
        'gamma': 0.995,
        # 'gamma': 0.75,
        'steps_per_cycle': 1,
        'train_per_cycle': 1,
        'use_lyapunov': True,
        'Time_near': True,
        'adaptive_alpha': True,
        'approx_value': True,
        'value_horizon': 2,
        # 'finite_horizon': True,
        'finite_horizon': False,
        'soft_predict_horizon': False,
        'target_entropy': None,
        'history_horizon': 0,  # 0 is using current state only
    },
    'EAC': {
        'iter_of_actor_train_per_epoch': 50,
        'iter_of_disturber_train_per_epoch': 50,
        'memory_capacity': int(1e6),
        'min_memory_size': 1000,
        'batch_size': 256,
        'labda': 1.,
        'alpha': 2.,
        'beta':0.1,
        'alpha3': 1,
        'tau': 5e-3,
        'lr_a': 1e-4,
        'lr_c': 3e-4,
        'lr_l': 3e-4,
        'gamma': 0.995,
        # 'gamma': 0.75,
        'steps_per_cycle': 1,
        'train_per_cycle': 1,
        'use_lyapunov': True,
        'Time_near': True,
        'adaptive_alpha': True,
        'approx_value': True,
        'value_horizon': 2,
        # 'finite_horizon': True,
        'finite_horizon': False,
        'soft_predict_horizon': False,
        'target_entropy': None,
        'history_horizon': 0,  # 0 is using current state only
    },
    'NAC': {
        'iter_of_actor_train_per_epoch': 50,
        'iter_of_disturber_train_per_epoch': 50,
        'memory_capacity': int(1e6),
        'min_memory_size': 1000,
        'batch_size': 256,
        'labda': 1.,
        'alpha': 2.,
        'beta': 0.5,
        'alpha3': 1,
        'theta': 0.5,
        'tau': 5e-3,
        'lr_a': 1e-4,
        'lr_c': 3e-4,
        'lr_l': 3e-4,
        'gamma': 0.995,
        # 'gamma': 0.75,
        'steps_per_cycle': 1,
        'train_per_cycle': 1,
        'use_lyapunov': True,
        'Time_near': True,
        'adaptive_alpha': True,
        'approx_value': True,
        'value_horizon': 2,
        # 'finite_horizon': True,
        'finite_horizon': False,
        'soft_predict_horizon': False,
        'target_entropy': None,
        'history_horizon': 0,  # 0 is using current state only
    },

    'CAC_Battery': {
        'iter_of_actor_train_per_epoch': 50,
        'iter_of_disturber_train_per_epoch': 50,
        'memory_capacity': int(1e6),
        'min_memory_size': 10000,
        'batch_size': 256,
        'labda': 1.,
        'alpha': 2.,
        'beta':0.,
        'alpha3': 1,
        'tau': 5e-3,
        'lr_a': 1e-4,
        'lr_c': 3e-4,
        'lr_l': 3e-4,
        'gamma': 0.995,
        # 'gamma': 0.75,
        'steps_per_cycle': 1,
        'train_per_cycle': 1,
        'use_lyapunov': True,
        'Time_near': True,
        'adaptive_alpha': True,
        'adaptive_beta': False,
        'approx_value': True,
        'value_horizon': 2,
        # 'finite_horizon': True,
        'finite_horizon': False,
        'soft_predict_horizon': False,
        'target_entropy': None,
        'history_horizon': 0,  # 0 is using current state only
    },
}

VARIANT = {
    'env_name' : 'BatteryCalib',
    'dataset_name':'data_50_trajectories_constant_load_8_uniform_q_7000_7600.npz',
    # 'env_name':'CMAPSS',
    #training prams
    'algorithm_name' : 'CAC_Battery',
    # 'algorithm_name': 'NAC',
     # 'algorithm_name': 'CAC',
    # 'algorithm_name': 'LAC',
    # 'additional_description': '_submit_policy_fixed_beta_0.5',
    # 'additional_description': '-new-reward',
    #  'additional_description': '-bias-0.01',
    # 'additional_description': '-dynamic-noisy-1',
    # 'additional_description': '-new-reward-0.01-noisy-training',
     'additional_description' : '-data_50_traj_const_W8_uni_q_7000_7600_reward1_{}-{}-{}-{}-{}-{}-{}'.format(
                                ALG_PARAMS['CAC_Battery']['labda'],
                                ALG_PARAMS['CAC_Battery']['alpha'],
                                ALG_PARAMS['CAC_Battery']['beta'],
                                ALG_PARAMS['CAC_Battery']['alpha3'],
                                ALG_PARAMS['CAC_Battery']['gamma'],
                                ALG_PARAMS['CAC_Battery']['adaptive_alpha'],
                                ALG_PARAMS['CAC_Battery']['adaptive_beta']
                                ),
    # 'evaluate': False,
     'train': True,
    # 'train': False, 
    # 'evaluate' : True,

    'num_of_trials': 1,   # number of random seeds

    'store_last_n_paths': 5,  # number of trajectories for evaluation during training
    'start_of_trial': 0,
    'eval_list': [

    ],
    'trials_for_eval': [str(i) for i in range(0, 10)],

    'evaluation_frequency': 25000,
}

VARIANT['log_path']='/'.join(['./log', VARIANT['env_name'], VARIANT['algorithm_name'] + VARIANT['additional_description']])





VARIANT['env_params']=ENV_PARAMS[VARIANT['env_name']]

VARIANT['alg_params']=ALG_PARAMS[VARIANT['algorithm_name']]

RENDER = True
def get_env_from_name(name):
    if name == 'CMAPSS':
        from envs.CMAPSS import CMAPSS as env
        env = env()
        env = env.unwrapped
    elif name == 'BatteryCalib':
        from envs.BatteryCalib import BatteryCalib as env
        env = env()
        env = env.unwrapped
    env.seed(SEED)
    return env

def get_train(name):
    if 'LAC' in name:
        from LAC.LAC import train
    elif 'CAC_Battery' in name:
        from LAC.CAC_Battery import train
    elif 'CAC' in name:
        from LAC.CAC import train
    elif 'EAC' in name:
        from LAC.EAC import train
    elif 'NAC' in name:
        from LAC.NAC import train

    return train

def get_policy(name):
    if 'CAC_Battery' in name:
        from LAC.CAC_Battery import CAC as build_func
    elif 'CAC' in name :
        from LAC.CAC import CAC as build_func
    elif 'LAC' in name :
        from LAC.LAC import LAC as build_func
    elif 'EAC' in name:
        from LAC.EAC import EAC as build_func
    elif 'NAC' in name:
        from LAC.NAC import NAC as build_func
    return build_func

def get_eval(name):
    if 'LAC' in name:
        from LAC.LAC import eval
    elif 'CAC_Battery' in name:
        from LAC.CAC_Battery import eval
    elif 'CAC' in name:
        from LAC.CAC import eval
    elif 'EAC' in name:
        from LAC.EAC import eval
    elif 'NAC' in name:
        from LAC.NAC import eval

    return eval


