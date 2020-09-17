import gym
import datetime
import numpy as np

SEED = 2020

ENV_PARAMS = {
    'BatteryCalib' : {
        'max_ep_steps' : int(16),
        'max_global_steps' : int(1000000),
        'max_episodes' : int(1000000),
        'eval_render': False,
        'action_low' : 5000,
        'action_high' : 8000,}
}

ALG_PARAMS = {
    'CAC': {
        'iter_of_actor_train_per_epoch': 50,
        'iter_of_disturber_train_per_epoch': 50,
        'memory_capacity': int(1e6),
        'min_memory_size': 1000,
        'batch_size': 128,
        'labda': 1.,
        'alpha': 2.,
        'beta':0.0,
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
        'value_horizon': 20,
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
        'lr_a': 5e-4,
        'lr_c': 5e-4,
        'lr_l': 5e-4,
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
    # 'dataset_name': 'data_5_trajectories_const_load_8_uniform_q_3000_7000_dt_1_short.npz',
    'dataset_name':'data_6_trajectories_uniform_load_8_16_uniform_q_6000_7000_dt_1_short.npz',
    'num_data_trajectories':6,
    'reward_id':2, # 1 to 6,
    'traj_start':"random",
    #training prams
    'algorithm_name' : 'CAC',
    'additional_description' : '-data_6_traj_uniform_load_8_16_uniform_q_6000_7000_dt_1_reward2_traj16_{}-{}-{}-{}-{}-{}-{}'.format(
    #  'additional_description' : '-data_5_traj_const_load_8_uniform_q_3000_7000_dt_1_reward6_short_fulltraj_{}-{}-{}-{}-{}-{}-{}'.format(
                                ALG_PARAMS['CAC']['labda'],
                                ALG_PARAMS['CAC']['alpha'],
                                ALG_PARAMS['CAC']['beta'],
                                ALG_PARAMS['CAC']['alpha3'],
                                ALG_PARAMS['CAC']['gamma'],
                                ALG_PARAMS['CAC']['adaptive_alpha'],
                                ALG_PARAMS['CAC']['adaptive_beta']
                                ),
    # 'evaluate': False,
    # 'train': True,
    'train': False, 
    'evaluate' : True,
    'num_of_trials': 1,   # number of random seeds
    'store_last_n_paths': 6,  # number of trajectories for evaluation during training
    'start_of_trial': 0,
    'eval_list': [
    ],
    'trials_for_eval': [str(i) for i in range(0, 10)],

    'evaluation_frequency': 10000,
}

VARIANT['log_path']='/'.join(['./log', VARIANT['env_name'], VARIANT['algorithm_name'] + VARIANT['additional_description']])

VARIANT['env_params']=ENV_PARAMS[VARIANT['env_name']]
VARIANT['alg_params']=ALG_PARAMS[VARIANT['algorithm_name']]
VARIANT['alg_params']['seed'] = SEED

RENDER = True
def get_env_from_name(name):
    if name == 'BatteryCalib':
        from envs.BatteryCalib import BatteryCalib as env
        env = env(action_low=ENV_PARAMS[name]['action_low'], action_high=ENV_PARAMS[name]['action_high'])
        env = env.unwrapped
    env.seed(SEED)
    return env

def get_train(name):

    if 'CAC_Battery' in name:
        from source.agent import train
    elif 'CAC' in name:
        from source.agent import train

    return train

def get_eval(name):

    if 'CAC_Battery' in name:
        from source.agent import eval
    elif 'CAC' in name:
        from source.agent import eval
    return eval


