"""A basic implementation of Generative Adversarial Imitation Learning using Stable Baselines 2.10.

- Author: Prabhasa Kalkur
- Note: choose {env, RL algo, training time, hyperparameters} from cmd line, # policy evaluations needed (line 94), # expert trajectories generated & used (line 101, 112)
"""

import os
import sys
import gym
import argparse
import warnings
import numpy as np
import tensorflow as tf

#Install Stable Baselines 2.10 (https://stable-baselines.readthedocs.io/en/master/guide/install.html)
from stable_baselines import GAIL, TRPO, SAC, DQN, PPO2, A2C, DDPG, ACER, ACKTR, HER, TD3
from stable_baselines.gail import ExpertDataset, generate_expert_traj
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.common import set_global_seeds
from stable_baselines.utils.utils import StoreDict

# Install AirSim 2.0 (https://pypi.org/project/airsim/ and https://microsoft.github.io/AirSim/build_windows/)
from airsim_env.envs.airsim_env_0 import AirSim

# numpy warnings because of tensorflow
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#Add your env and algo here. Define your reward criteria in env_success
algo_list = {'sac': SAC, 'trpo': TRPO, 'gail': GAIL, 'dqn': DQN, 'ppo2': PPO2,
            'ddpg': DDPG, 'acer': ACER, 'acktr': ACKTR, 'her': HER, 'td3': TD3}
env_list = ['Pendulum-v0', 'CartPole-v1', 'LunarLander-v2', 'LunarLanderContinuous-v2', 'BipedalWalker-v2', 'BipedalWalkerHardcore-v2',
            'HalfCheetah-v2', 'Hopper-v2', 'Humanoid-v2', 'Ant-v2', 'Reacher-v2', 'Swimmer-v2', 'AirSim-v0'] #mujoco envs, needs license
env_success = [-200, 475, 200, 200, 300, 300, 9000, 3000, 1000, 6000, 3.75, 360, 1000] #not exactly what OpenAI Gym defines

def train(exp_id, env_id, env, algo, policy, seed, timesteps, hyperparams=None, dataset=None):
    env_name = env_id[:-3]
    if algo!='gail':
        model = (algo_list[algo])(policy=policy, env=env, seed=seed, n_cpu_tf_sess=1, **hyperparams) #n_cpu_tf_sess=1 for deterministic CPU results
        model.learn(total_timesteps=timesteps)
        model.save("models/{}_{}_{}".format(env_name.lower(), algo.lower(), exp_id)) #e.g. trpo_hopper_0
    elif algo=='gail':
        model = (algo_list[algo])(policy=policy, env=env, seed=seed, expert_dataset=dataset, n_cpu_tf_sess=1, **hyperparams)
        model.learn(total_timesteps=timesteps)
        model.save("models/gail_{}_{}_{}".format(env_name.lower(), algo.lower(), exp_id)) #e.g. gail_trpo_hopper_0

def evaluate(mode, quantity, env_id, env, algo, n_eval_episodes):
    env_index = env_list.index(env_id)
    if mode=='policy':
        print('\nTrained {} with {}. Some stats:\n'.format(env_id, algo))
        episode_return, episode_length = evaluate_policy(quantity, env, n_eval_episodes=n_eval_episodes, return_episode_rewards=True)
        success_count = sum(i >= env_success[env_index] for i in episode_return)
        success_criteria = episode_return
    elif mode=='expert':
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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='environment ID', type=str, default='CartPole-v1', choices=env_list)
    parser.add_argument('--algo', help='RL Algorithm', default='trpo', type=str, required=False, choices=list(algo_list.keys()))
    parser.add_argument('--exp-id', help='Experiment ID', default=0, type=int)
    parser.add_argument('--seed', help='Random generator seed', type=int, default=0)

    parser.add_argument('-rl', '--train-RL', help='To train RL', action='store_true')
    parser.add_argument('-trl', '--timesteps-RL', help='Number of timesteps for RL', default="1e5", type=str)
    parser.add_argument('-il', '--train-IL', help='To train GAIL', action='store_true')
    parser.add_argument('-til', '--timesteps-IL', help='Number of timesteps for GAIL', default="1e6", type=str)

    parser.add_argument('-params-RL', '--hyperparams-RL', type=str, nargs='+', default={}, action=StoreDict, help='Overwrite hyperparameter (e.g. gamma:0.95 timesteps_per_batch: 2048)')
    parser.add_argument('-params-IL', '--hyperparams-IL', type=str, nargs='+', default={}, action=StoreDict, help='Overwrite hyperparameter (e.g. gamma:0.95 timesteps_per_batch: 2048)')
    args = parser.parse_args()
    return args

def main():

    # Initialize env (for vectorized env, use DummyVecEnv([make_env(env_id, 0, args.seed)]))
    args = get_args()
    env_id, algo, exp_id = args.env, args.algo, args.exp_id
    env_name = env_id[:-3]
    env = gym.make(env_id)
    env.seed(args.seed)
    set_global_seeds(args.seed)

    folder_RL = [env_name.lower(), args.algo.lower(), exp_id]
    folder_IL = [env_name.lower(), 'gail', exp_id]
    policy_RL = 'MlpPolicy'
    policy_IL = 'MlpPolicy'
    eval_episodes = 100 #evaluation episodes for trained model

    # Train RL algo on env
    if args.train_RL:
        train(exp_id, env_id, env, algo, policy_RL, args.seed, timesteps=int(float(args.timesteps_RL)), hyperparams=args.hyperparams_RL)
    
        # Generate expert trajectories
        expert_traj_gen = 100
        model = (algo_list[algo]).load("models/{}_{}_{}".format(*folder_RL))
        model.set_env(env)
        generate_expert_traj(model, env=env, save_path='experts/{}_{}_{}'.format(*folder_RL), n_episodes=expert_traj_gen) #comment lines 172, 173 in function to not print expert data info

    # Evaluate RL model

    if os.path.exists("models/{}_{}_{}.zip".format(*folder_RL)):
        model = (algo_list[algo]).load("models/{}_{}_{}".format(*folder_RL))
        evaluate('policy', model, env_id, env, algo, eval_episodes)
    else:
        print('RL model unavailable. Please train...')

    #  Load expert dataset
    if os.path.exists('experts/{}_{}_{}.npz'.format(*folder_RL)):
        expert_traj_use = 10 # using only 10 trajectories
        dataset = ExpertDataset(expert_path='experts/{}_{}_{}'.format(*folder_RL), traj_limitation=expert_traj_use, verbose=0)
        expert_data = np.load('{}_{}_{}.npz'.format(*folder_RL), allow_pickle =True)
        evaluate('expert', expert_data, env_id, env, algo, expert_traj_use)
    else:
        print('Expert data unavailable. Please generate...')
        sys.exit()

    # Train GAIL on expert
    if args.train_IL:
        train(exp_id, env_id, env, 'gail', policy_IL, args.seed, timesteps=int(float(args.timesteps_IL)), hyperparams=args.hyperparams_IL, dataset=dataset)

    # Evaluate IL model
    if os.path.exists("models/{}_{}_{}.zip".format(*folder_IL)):
        model = (algo_list['gail']).load("models/gail_{}_{}_{}".format(*folder_IL))
        evaluate('policy', model, env_id, env, algo, eval_episodes)
    else:
        print('GAIL model unavailable. Please train...')

if __name__ == "__main__":
    main()