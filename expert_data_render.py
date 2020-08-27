# Render custom .npz expert data files from Stable Baselines 2.10 by creating CustomEnvs of Gym envs
# Author: Prabhasa Kalkur

import os
import sys
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

from custom_env.envs import PendulumEnv, CartPoleEnv, LunarLander, LunarLanderContinuous
from stable_baselines import GAIL, SAC, TRPO, DQN, PPO2, A2C, DDPG, ACER, ACKTR, HER, TD3
from stable_baselines.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines.common import set_global_seeds
from utils import make_env, StoreDict

# Please choose env accordingly
from airsim_env.envs.airsim_env_0 import AirSim # 'AirSim-v0'
# from expert_render.envs.airsim_env_0 import AirSim # 'My-AirSim-v0'

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
done_count, success_count, episode_reward, total_reward = 0, 0, 0, 0

algo_list = {'sac': SAC, 'trpo': TRPO, 'gail': GAIL, 'dqn': DQN, 'ppo2': PPO2,
            'ddpg': DDPG, 'a2c': A2C, 'acktr': ACKTR, 'her': HER, 'td3': TD3}
env_list_custom = ['My-Pendulum-v0', 'My-CartPole-v1', 'My-LunarLander-v2', 'My-LunarLanderContinuous-v2', 'My-AirSim-v0', 'AirSim-v0']
env_success = [-200, 475, 200, 200, 1000] # OpenAI Gym requirements (Hopper should be 3800)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', help='CPU or GPU', default='cpu', type=str, choices=['cpu', 'gpu'])
    parser.add_argument('--seed', help='Random generator seed', type=int, default=0)
    parser.add_argument('--env', help='environment ID', type=str, default='Pendulum-v0', choices=env_list_custom)
    parser.add_argument('--algo', help='RL Algorithm', default='trpo', type=str, required=False, choices=list(algo_list.keys()))
    
    parser.add_argument('--exp-id', help='Experiment ID', default=0, type=int)
    parser.add_argument('-norm', '--normalize', help='Normalize data', action='store_true')
    parser.add_argument('-human', '--human-expert', help='If using human expert', action='store_true')
    parser.add_argument('--render', help='Render expert data', action='store_true')
    parser.add_argument('--episodic', help='Render expert data', action='store_true')
    parser.add_argument('--env-kwargs', type=str, nargs='+', action=StoreDict, help='Optional keyword argument to pass to the env constructor')
    
    args = parser.parse_args()
    return args

def evaluate(mode, quantity, env_id, env, algo, n_eval_episodes):
    env_index = env_list_custom.index(env_id)
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

def check_success(env_index, env_success, success_count):
    if episode_reward>=env_success[env_index]:
        success_count+=1
        # print('Test: Success!')
    else:
        # print('Test: Fail!')
        pass
    return success_count

def choose_device(device_name):
    if device_name == "gpu":
        device_name = "/gpu:0"
    else:
        device_name = "/cpu:0"

def main():
    global done_count, success_count, episode_reward, total_reward
    args = get_args()
    set_global_seeds(args.seed)
    choose_device(args.device)
    env_id, algo, exp_id = args.env, args.algo, args.exp_id
    env_index = env_list_custom.index(env_id)
    env_name = env_id[3:-3] #note [3:]
    folder = [exp_id, env_name.lower(), algo.lower()]
    
    env_kwargs = {} if args.env_kwargs is None else args.env_kwargs # If AirSim-v0 reward is not 1000, please change env_success too!
    # env = DummyVecEnv([make_env(env_id, 0, args.seed, env_kwargs=env_kwargs)]) # CANNOT HAVE A DUMMYVEC AS IT HAS ITS OWN RESET
    env = gym.make(env_id, **env_kwargs)
    env.seed(args.seed)

    if args.normalize:
        callback_path =  os.path.join('callbacks', "{}_{}_{}".format(*folder), 'normalized_obs')
        env = VecNormalize.load(os.path.join(callback_path, "vec_normalize.pkl"), env)
        env.training = False
        env.norm_reward = False
        env.seed(args.seed)

    print('Here are some stats of the expert... ')
    if args.human_expert and (env_id in ['AirSim-v0', 'My-AirSim-v0']):
        restrict = env_kwargs['restrict'] if ('restrict' in env_kwargs) else True
        reward = int(env_kwargs['rew_land']) if ('rew_land' in env_kwargs) else 1000
        env_success[-1] = reward
        expert_data = np.load(os.path.join('expert', 'human', 'expert_data_140_soft_sample_25_norm_{}_restrict_space_{}_success_{}_scale_2_v_0.npz'.format(args.normalize, restrict, reward)), allow_pickle =True)
    else:
        if args.normalize:
            expert_data = np.load('experts/{}_{}_{}_norm_obs.npz'.format(*folder), allow_pickle=True) #Gym envs
        else:
            expert_data = np.load('experts/{}_{}_{}.npz'.format(*folder), allow_pickle=True) #Gym-envs
    evaluate('expert', expert_data, env_id, None, algo, 100)

    # For rendering expert, episode-wise
    if args.render:
        states = expert_data['obs']
        actions = expert_data['actions']
        returns = expert_data['episode_returns']

        if env_id in ['My-Pendulum-v0']:
            states = np.stack((np.arccos(states[:,0]), states[:,2]), axis=-1)

        env.reset(states[0])
        for i in range(len(expert_data['obs'])):
            action = actions[i]
            if env_id in ['My-CartPole-v1', 'My-LunarLander-v2']: #Need to unpack discrete space value
                env.step(action[0], states[i+1])
            elif env_id in ['AirSim-v0']:
                env.step(action, states[i], states[i+1])
            else:
                env.step(action, states[i+1])
            # rewards = expert_data['rewards'][i]
            if expert_data['episode_starts'][i+1]:
                dones = True
            else:
                dones = False
            env.render()
            if dones:
                if args.episodic:
                    import pdb; pdb.set_trace() # Episodic view of expert data
                done_count+=1
                print('episode_reward: ', returns[done_count-1])
                success_count = check_success(env_index, env_success, success_count)
                total_reward+=episode_reward
                episode_reward = 0
                env.reset(states[i+1])
        print('\n{}/{} successful episodes'.format(success_count, done_count))
        average_reward = total_reward/done_count
        print('\nAverage reward: {}'.format(average_reward))
        env.close()

if __name__ == '__main__':
    main()

#Alternate, smaller version of test:
# if args.test:
    # counter=0
    # while True:
    #     counter+=1
    #     env.step(expert_data['actions'][counter])
    #     env.render()
    #     if expert_data['episode_starts'][counter+1]:
    #         env.reset()
    #     else:
    #         continue