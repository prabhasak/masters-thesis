# Evaluate and test trained RL/IL model from Stable Baselines 2.10
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

from stable_baselines import GAIL, SAC, TRPO, DQN, PPO2, A2C, DDPG, ACER, ACKTR, HER, TD3
from stable_baselines.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.common import set_global_seeds

from utils import StoreDict, make_env
from airsim_env.envs.airsim_env_0 import AirSim

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
done_count, success_count, episode_reward, total_reward = 0, 0, 0, 0

algo_list = {'sac': SAC, 'trpo': TRPO, 'gail': GAIL, 'dqn': DQN, 'ppo2': PPO2,
            'ddpg': DDPG, 'a2c': A2C, 'acktr': ACKTR, 'her': HER, 'td3': TD3}
env_list = ['Pendulum-v0', 'CartPole-v1', 'LunarLander-v2', 'LunarLanderContinuous-v2', 'MountainCarContinuous-v0', 'BipedalWalker-v3',
            'HalfCheetah-v2', 'Hopper-v2', 'Humanoid-v2', 'Ant-v2', 'Reacher-v2', 'Swimmer-v2', 'AirSim-v0'] # mujoco envs need license
env_success = [-200, 475, 200, 200, 90, 300, 4800, 3000, 1000, 6000, 3.75, 360, 1000] # OpenAI Gym requirements (Hopper should be 3800)
episode_len = [200, 500, 400, 400, 999, 1600, 1000, 1000, 1000, 1000, 50, 1000, 100]

def get_args():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', help='CPU or GPU', default='cpu', type=str, choices=['cpu', 'gpu'])
    parser.add_argument('--seed', help='Random generator seed', type=int, default=0)
    parser.add_argument('--env', help='environment ID', type=str, default='Pendulum-v0', choices=env_list)
    parser.add_argument('--algo', help='RL Algorithm', default='trpo', type=str, required=False, choices=list(algo_list.keys()))
    parser.add_argument('--exp-id', help='Experiment ID', default=0, type=int)

    parser.add_argument('--mode', help='Choose between RL or IL', type=str, default='rl', choices=['rl', 'il'])
    parser.add_argument('-policy', '--evaluate-policy', help='Evaluate trained policy', action='store_true')
    parser.add_argument('-norm', '--normalize', help='Normalize data', action='store_true')
    parser.add_argument('--test', help='Test trained policy', action='store_true')
    parser.add_argument('--env-kwargs', type=str, nargs='+', action=StoreDict, help='Optional keyword argument to pass to the env constructor')
    
    args = parser.parse_args()
    return args

def evaluate(mode, quantity, env_id, env, algo, n_eval_episodes):
    env_index = env_list.index(env_id)
    if mode=='policy':
        episode_return, episode_length = evaluate_policy(quantity, env, n_eval_episodes=n_eval_episodes, return_episode_rewards=True)
        success_count = sum(i >= env_success[env_index] for i in episode_return)
        success_criteria = episode_return
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
        print('Test: Success!')
    else:
        print('Test: Fail!')
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
    env_index = env_list.index(env_id)
    env_name = env_id[:-3]

    env_kwargs = {} if args.env_kwargs is None else args.env_kwargs
    env = DummyVecEnv([make_env(env_id, 0, args.seed, env_kwargs=env_kwargs)])

    folder = [exp_id, env_name.lower(), algo.lower()]


    # Pass CustomEnv arguments: follow this for your CustomEnv if reward not known prior to training
    env_kwargs = {} if args.env_kwargs is None else args.env_kwargs
    if (args.env_kwargs is not None) and (env_id in ['AirSim-v0']):
        if 'rew_land' in env_kwargs:
            if (int(env_kwargs['rew_land']) in [500, 1000, 10000]):
                env_success[-1] = int(env_kwargs['rew_land'])
            else:
                raise ValueError('Given env reward not acceptable. Please try again') 

    # Load environment
    env = DummyVecEnv([make_env(env_id, 0, args.seed, env_kwargs=env_kwargs)])
    if (args.normalize and (not args.human_expert)):
        if args.mode=='il':
            callback_path =  os.path.join("callbacks/{}_{}_{}_gail".format(*folder), 'normalized_obs')
            env = VecNormalize.load(os.path.join(callback_path, "vec_normalize_gail_old.pkl"), env) #change to new if needed
        else:
            callback_path =  os.path.join("callbacks/{}_{}_{}".format(*folder), 'normalized_obs')
            env = VecNormalize.load(os.path.join(callback_path, "vec_normalize_rl_old.pkl"), env) #change to new if needed
        env.training = False
        env.norm_reward = False
        env.seed(args.seed)

    # Load model
    if args.mode=='il':
        algorithm = algo_list['gail']
        model_path = "models/{}_{}_{}_gail_norm_obs".format(*folder) if args.normalize else "models/{}_{}_{}_gail".format(*folder)
        print('Here are some stats of the GAIL model... ')
    elif args.mode=='rl':
        algorithm = algo_list[algo]
        model_path = "models/{}_{}_{}_norm_obs".format(*folder) if args.normalize else "models/{}_{}_{}".format(*folder)
        print('Here are some stats of the RL model... ')
    model = algorithm.load(model_path)
    model.set_env(env)

    if args.evaluate_policy:
        evaluate('policy', model, env_id, env, algo, 100)

    if args.test:
        print('\nTesting policy...')
        n_timesteps = 10*episode_len[env_index]
        obs = env.reset()
        for _ in range(n_timesteps):
            action, _states = model.predict(obs, deterministic=True)
            if isinstance(env.action_space, gym.spaces.Box):
                action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, rewards, dones, info = env.step(action)
            episode_reward+=rewards
            env.render()
            if dones:
                done_count+=1
                success_count = check_success(env_index, env_success, success_count)
                total_reward+=episode_reward
                episode_reward = 0
                env.reset()
        print('\n{}/{} successful episodes'.format(success_count, done_count))
        average_reward = total_reward/done_count
        print('\nAverage reward: {}'.format(average_reward))
        env.close()

if __name__ == '__main__':
    main()

# Alternate, smaller version of test:
# if args.test:
    # obs = env.reset()
    # for _ in range(n_timesteps):
    #     action, _ = model.predict(obs, deterministic=True)
    #     obs, rewards, dones, info = env.step(action)
    #     env.render()
    #     if dones:
    #         env.reset()
    # env.close()
