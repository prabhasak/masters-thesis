"""Benchmark reinforcement learning (RL) and imitation Learning (GAIL) algorithms from Stable Baselines 2.10.
Author: Prabhasa Kalkur

- Note 1.0: choose {env, RL algo, training times, hyperparameters, } as cmd line arguments
- Note 1.1: please choose appropriate policy for the RL/IL algorithms (line 151, 152)       
- Note 1.2: changeable numbers in the program:
            number of episodes used for policy evaluation after training = 100 (line 153)
            number of expert trajectories generated & used = 100, 10 (line 153)
            callback model saving and evaluation = every 100, 300 episodes for RL, IL (174, 205)
- Note 2: Things you can add on top: Multiprocessing, Monitor, VecNormalize, HP tuning, pass CustomEnv kwargs
"""

try:
    import mpi4py
    from mpi4py import MPI
except ImportError:
    mpi4py = None
    
import os
import sys
import gym
import argparse
import warnings
import numpy as np
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#Install Stable Baselines 2.10 (https://stable-baselines.readthedocs.io/en/master/guide/install.html)
from stable_baselines import GAIL, TRPO, SAC, DQN, PPO1, A2C, DDPG, ACER, ACKTR, HER, TD3
from stable_baselines.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines.gail import ExpertDataset, generate_expert_traj
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.common import set_global_seeds
from utils import StoreDict
from shutil import copy

# Install AirSim 2.0 (https://pypi.org/project/airsim/ and https://microsoft.github.io/AirSim/build_windows/)
from airsim_env.envs.airsim_env_0 import AirSim

# Add your coustom env and algo details here. Reference: https://github.com/openai/gym/blob/master/gym/envs/__init__.py
algo_list = {'sac': SAC, 'trpo': TRPO, 'gail': GAIL, 'dqn': DQN, 'ppo1': PPO1,
            'ddpg': DDPG, 'a2c': A2C, 'acktr': ACKTR, 'her': HER, 'td3': TD3}
env_list = ['Pendulum-v0', 'CartPole-v1', 'LunarLander-v2', 'LunarLanderContinuous-v2', 'MountainCarContinuous-v0', 'BipedalWalker-v3',
            'HalfCheetah-v2', 'Hopper-v2', 'Humanoid-v2', 'Ant-v2', 'Reacher-v2', 'Swimmer-v2', 'AirSim-v0'] # mujoco envs need license
env_success = [-200, 475, 200, 200, 90, 300, 4800, 3000, 1000, 6000, 3.75, 360, 1000] # OpenAI Gym requirements (Hopper should be 3800)
episode_len = [200, 500, 400, 400, 999, 1600, 1000, 1000, 1000, 1000, 50, 1000, 100]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', help='CPU or GPU', default='cpu', type=str, choices=['cpu', 'gpu'])
    parser.add_argument('--seed', help='Random generator seed', type=int, default=0)
    parser.add_argument('--env', help='environment ID', type=str, default='CartPole-v1', choices=env_list)
    parser.add_argument('--algo', help='RL Algorithm', default='trpo', type=str, required=False, choices=list(algo_list.keys()))
    parser.add_argument('--exp-id', help='Experiment ID', default=0, type=int)
    parser.add_argument('--verbose', help='Verbose mode (0: no output, 1: INFO)', default=0, type=int)

    parser.add_argument('-rl', '--train-RL', help='To train RL', action='store_true')
    parser.add_argument('-trl', '--timesteps-RL', help='Number of timesteps for RL', default="1e5", type=str)
    parser.add_argument('-il', '--train-IL', help='To train GAIL', action='store_true')
    parser.add_argument('-til', '--timesteps-IL', help='Number of timesteps for GAIL', default="1e6", type=str)
    parser.add_argument('-best', '--save-best-model', help='For saving best model from EvalCallback instead of last available model', action='store_true')
    
    parser.add_argument('-check', '--check-callback', help='For saving models every save_freq steps', action='store_true')
    parser.add_argument('-eval', '--eval-callback', help='For evaluating model every eval_freq steps', action='store_true')
    parser.add_argument('-tb', '--tensorboard', help='For Tensorboard logging', action='store_true')
    parser.add_argument('-params-RL', '--hyperparams-RL', type=str, nargs='+', default={}, action=StoreDict, help='Overwrite hyperparameter (e.g. gamma:0.95 timesteps_per_batch: 2048)')
    parser.add_argument('-params-IL', '--hyperparams-IL', type=str, nargs='+', default={}, action=StoreDict, help='Overwrite hyperparameter (e.g. gamma:0.95 timesteps_per_batch: 2048)')
    args = parser.parse_args()
    return args

def train(mode, save_best_model, folder, env, algo, policy, seed, timesteps, verbose=0, callback = None, tensorboard_log=None, hyperparams=None, dataset=None):
    if mode=='RL':
        model = (algo_list[algo])(policy=policy, env=env, seed=seed, n_cpu_tf_sess=1, tensorboard_log=tensorboard_log, verbose=verbose, **hyperparams) #n_cpu_tf_sess=1 for deterministic CPU results
        model.learn(total_timesteps=timesteps, callback = callback)
        if not save_best_model:
            model.save("models/{}_{}_{}".format(*folder)) #e.g. 0_trpo_hopper
    elif mode=='IL':
        model = (algo_list['gail'])(policy=policy, env=env, seed=seed, expert_dataset=dataset, n_cpu_tf_sess=1, tensorboard_log=tensorboard_log, **hyperparams)
        model.learn(total_timesteps=timesteps, callback = callback)
        if not save_best_model:
            model.save("models/{}_{}_{}_gail".format(*folder)) #e.g. 0_trpo_hopper_gail

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

def add_callback(callback, mode, env, folder, checkpoints, evaluations, save_freq, eval_freq):
    if mode=='RL':
        callback_path = "callbacks/{}_{}_{}".format(*folder)
    elif mode=='IL':
        callback_path = "callbacks/{}_{}_{}_gail".format(*folder)
    make_dir(callback_path)
    if checkpoints:
        callback.append(CheckpointCallback(save_freq=save_freq, save_path=callback_path, name_prefix='rl_model', verbose=1))
    if evaluations:
        callback.append(EvalCallback(env, best_model_save_path=callback_path, log_path=callback_path, eval_freq=eval_freq, verbose=1))
    return callback

def copy_best_model(mode, folder):
    print('\n\nChoosing best saved model throughout training, instead of model available at end of training (EvalCallback enabled!)')
    if mode=='RL':
        best_model_path = "callbacks/{}_{}_{}".format(*folder)
        last_model_path = "models/{}_{}_{}".format(*folder)
    elif mode=='IL':
        best_model_path = "callbacks/{}_{}_{}_gail".format(*folder)
        last_model_path = "models/{}_{}_{}_gail".format(*folder)
    copy(os.path.join(best_model_path,'best_model.zip'), "models/")
    if os.path.exists(last_model_path+'.zip'):
        os.remove(last_model_path+'.zip')
    os.rename("models/best_model.zip", last_model_path+'.zip')

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

def choose_device(device_name):
    if device_name == "gpu":
        device_name = "/gpu:0"
    else:
        device_name = "/cpu:0"

def main():

    # Initialize env (for vectorized env, use DummyVecEnv([make_env(env_id, 0, args.seed)]))
    args = get_args()
    set_global_seeds(args.seed)
    choose_device(args.device)
    env_id, algo, exp_id = args.env, args.algo, args.exp_id
    env_index = env_list.index(env_id) # for callbacks
    env_name = env_id[:-3]
    env = gym.make(env_id)
    env.seed(args.seed)

    folder = [exp_id, env_name.lower(), args.algo.lower()]
    policy_RL = 'MlpPolicy'
    policy_IL = 'MlpPolicy'
    policy_eval_episodes, expert_traj_gen, expert_traj_use = 100, 100, 10
    save_freq_RL, eval_freq_RL, save_freq_IL, eval_freq_IL, save_best_model, tensorboard_path = 0, 0, 0, 0, False, None # default tensorboard, callback values

    # checking arguments for correctness
    if not ((args.train_RL) or (args.train_IL)):
        print("This program is for training an RL/IL algorithm. PLease specify at least one of the options...")
        sys.exit()

    # Adding Tensorboard features
    if args.tensorboard:
        tensorboard_path = "tensorboard/{}_{}".format(exp_id, env_name.lower())
        make_dir(tensorboard_path)

    # Adding callback features
    if (args.eval_callback and args.save_best_model):
        save_best_model = True

    ############  BODY OF THE CODE: RL -> EXPERT -> GAIL  ##############

    # Train RL algo on env
    if args.train_RL:
        if (args.check_callback or args.eval_callback):
            save_freq_RL, eval_freq_RL = 100*episode_len[env_index], 100*episode_len[env_index]
            callback = add_callback([], 'RL', env, folder, args.check_callback, args.eval_callback, save_freq_RL, eval_freq_RL)
        train('RL', save_best_model, folder, env, algo, policy_RL, args.seed, timesteps=int(float(args.timesteps_RL)), verbose=args.verbose,
            tensorboard_log=tensorboard_path, callback = callback, hyperparams=args.hyperparams_RL)
        if (args.eval_callback and args.save_best_model):
            copy_best_model('RL', folder)

        # Evaluate RL model - inside because it is not crucial to the problem
        model = (algo_list[algo]).load("models/{}_{}_{}".format(*folder))
        evaluate('policy', model, env_id, env, algo, policy_eval_episodes)

    # Generate expert trajectories
    if os.path.exists("models/{}_{}_{}.zip".format(*folder)):
        model = (algo_list[algo]).load("models/{}_{}_{}".format(*folder))
        model.set_env(env)
        generate_expert_traj(model, env=env, save_path='experts/{}_{}_{}'.format(*folder), n_episodes=expert_traj_gen) #comment lines 172, 173 in function to not print expert data info
    else:
        print("Trained RL model unavailable. Please train again...")

    #  Load expert dataset
    if os.path.exists('experts/{}_{}_{}.npz'.format(*folder)):
        dataset = ExpertDataset(expert_path='experts/{}_{}_{}.npz'.format(*folder), traj_limitation=expert_traj_use, verbose=0)
        expert_data = np.load('experts/{}_{}_{}.npz'.format(*folder), allow_pickle=True)
        evaluate('expert', expert_data, env_id, env, algo, expert_traj_gen)
    else:
        print('Expert data unavailable. Please generate...')
        sys.exit()

    # Train GAIL on expert
    if args.train_IL:
        if (args.check_callback or args.eval_callback):
            save_freq_IL, eval_freq_IL = 300*episode_len[env_index], 300*episode_len[env_index]
            callback = add_callback([], 'IL', env, folder, args.check_callback, args.eval_callback, save_freq_IL, eval_freq_IL)
        train('IL', save_best_model, folder, env, algo, policy_IL, args.seed, timesteps=int(float(args.timesteps_IL)), verbose=args.verbose,
            tensorboard_log=tensorboard_path, callback = callback, hyperparams=args.hyperparams_IL, dataset=dataset)
        if (args.eval_callback and args.save_best_model):
            copy_best_model('IL', folder)

    # Evaluate GAIL model
    if os.path.exists("models/{}_{}_{}_gail.zip".format(*folder)):
        model = (algo_list['gail']).load("models/{}_{}_{}_gail".format(*folder))
        evaluate('policy', model, env_id, env, algo, policy_eval_episodes)
    else:
        print('GAIL model unavailable. Please train...')

if __name__ == "__main__":
    main()