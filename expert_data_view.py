# View .npz expert data files from Stable Baselines 2.10
# Author: Prabhasa Kalkur

import os
import sys
import csv
import random
import numpy as np
import pandas as pd
import pickle as pkl

import argparse
import warnings
import tensorflow as tf

try:
    import mpi4py
    from mpi4py import MPI
except ImportError:
    mpi4py = None

import gym
import numpy as np
import stable_baselines

from stable_baselines.common import set_global_seeds
from stable_baselines import GAIL, SAC, TRPO, DQN, PPO2, A2C, DDPG, ACER, ACKTR, HER, TD3
from airsim_env.envs.airsim_env_0 import AirSim
from utils import StoreDict

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

algo_list = {'sac': SAC, 'trpo': TRPO, 'gail': GAIL, 'dqn': DQN, 'ppo2': PPO2,
            'ddpg': DDPG, 'a2c': A2C, 'acktr': ACKTR, 'her': HER, 'td3': TD3}
env_list = ['Pendulum-v0', 'CartPole-v1', 'LunarLander-v2', 'LunarLanderContinuous-v2', 'MountainCarContinuous-v0', 'BipedalWalker-v3',
            'HalfCheetah-v2', 'Hopper-v2', 'Humanoid-v2', 'Ant-v2', 'Reacher-v2', 'Swimmer-v2', 'AirSim-v0'] # mujoco envs need license
env_success = [-200, 475, 200, 200, 90, 300, 4800, 3000, 1000, 6000, 3.75, 360, 1000] # OpenAI Gym requirements (Hopper should be 3800)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', help='CPU or GPU', default='cpu', type=str, choices=['cpu', 'gpu'])
    parser.add_argument('--seed', help='Random generator seed', type=int, default=0)
    parser.add_argument('--env', help='environment ID', type=str, default='Pendulum-v0', choices=env_list)
    parser.add_argument('--algo', help='RL Algorithm', default='trpo', type=str, required=False, choices=list(algo_list.keys()))

    parser.add_argument('--exp-id', help='Experiment ID', default=0, type=int)
    parser.add_argument('-human', '--human-expert', help='If using human expert', action='store_true')
    parser.add_argument('-norm', '--normalize', help='Normalize data', action='store_true')

    parser.add_argument('--complete', help='Complete data', action='store_true')
    parser.add_argument('--episodic', help='Episodic data ', action='store_true')
    parser.add_argument('--env-kwargs', type=str, nargs='+', action=StoreDict, help='Optional keyword argument to pass to the env constructor')

    args = parser.parse_args()
    return args

def evaluate(mode, quantity, env_id, env, env_success, algo, n_eval_episodes):
    env_index = env_list.index(env_id)
    if mode=='expert':
        print('\nGenerated {} expert with {} policy. Some stats:\n'.format(env_id, algo))
        episode_return, episode_length = quantity['episode_returns'], len(quantity['obs'])/len(quantity['episode_returns'])
        success_count = sum(i >= env_success[env_index] for i in quantity['episode_returns'])
        success_criteria = quantity['episode_returns']
    print('{}/{} successful episodes\n'.format(success_count, n_eval_episodes))
    print('Mean return: ', np.mean(episode_return))
    print('Std return: ', np.std(episode_return))
    print('Max return: ', max(episode_return))
    print('Min return: ', min(episode_return))
    print('Mean episode len: ', np.rint(np.mean(episode_length)))
    if np.mean(success_criteria) >= env_success[env_index]:
        print('Optimal {} found!\n'.format(mode))
    else:
        print('Suboptimal {}. Please try again...'.format(mode))
        sys.exit()

def choose_device(device_name):
    if device_name == "gpu":
        device_name = "/gpu:0"
    else:
        device_name = "/cpu:0"

def main():
    args = get_args()
    set_global_seeds(args.seed)
    choose_device(args.device)
    env_id, algo, exp_id = args.env, args.algo, args.exp_id
    env_name = env_id[:-3]

    folder = [exp_id, env_name.lower(), algo.lower()]
    env_kwargs = {} if args.env_kwargs is None else args.env_kwargs

    print('Here are some stats of the expert... ')
    if args.human_expert and (env_id in ['AirSim-v0']):
        restrict = True if env_kwargs['restrict'] is None else env_kwargs['restrict']
        reward = 1000 if env_kwargs['rew_land'] is None else int(env_kwargs['rew_land'])
        env_success[-1] = reward
        expert_data = np.load(os.path.join('expert', 'human', 'expert_data_140_soft_sample_25_norm_{}_restrict_space_{}_success_{}_scale_2_v_0.npz'.format(args.normalize, restrict, reward)), allow_pickle =True)
    else:
        if args.normalize:
            expert_data = np.load('experts/{}_{}_{}_norm_obs.npz'.format(*folder), allow_pickle=True) #Gym envs
        else:
            expert_data = np.load('experts/{}_{}_{}.npz'.format(*folder), allow_pickle=True) #Gym-envs
    evaluate('expert', expert_data, env_id, None, env_success, algo, 100)

    #Complete view of expert data: prints complete data and max-min values (not recommended)
    if args.complete:
        print('Here is a complete view of the expert data with some stats...')
        for keys in expert_data:
            print('\nkey: ', keys)
            # print('data: ', expert_data[keys]) # enable to see data
            print('max value: ', np.amax(expert_data[keys], axis=0))
            print('min value: ', np.amin(expert_data[keys], axis=0))
            print('shape: ', expert_data[keys].shape)

    #Episodic view of expert data: choose component of data to view
    if args.episodic:
        print('Here is an episodic view of expert data. Please use "c" to parse through each episode and "q" to quit')
        indices = np.append(np.where(expert_data['episode_starts']==1)[0], len(expert_data['obs']+1))
        # print(indices)
        for i in range(len(expert_data['episode_returns'])):
            import pdb; pdb.set_trace()
            print('\nEpisode {}: '.format(i))
            # print('states: ', expert_data['obs'][indices[i]:indices[i+1]]) # "Useful" data in AirSim [:, 0:3], Pendulum[:, 0:1]
            # print(expert_data['obs'][indices[i]:indices[i+1]].shape)
            # print('actions: ', expert_data['actions'][indices[i]:indices[i+1]])
            # print(expert_data['actions'][indices[i]:indices[i+1]].shape)
            print('rewards: ', expert_data['rewards'][indices[i]:indices[i+1]], end=' ')
            print(expert_data['rewards'][indices[i]:indices[i+1]].shape)
            print('episode return: ', expert_data['episode_returns'][i])

if __name__ == '__main__':
    main()

# .pkl for OpenAI examples
# if __name__ == '__main__':
    # algo = sys.argv[1]
    # env = sys.argv[2]
    # expert_data = 'expert/{}.pkl'.format(algo, env_id)
    # with open(expert_data, 'rb') as f:
    #     data = pkl.load(f)
    # for keys in data:
        # print(type(data[keys]))
        # print(np.shape(data[keys]))