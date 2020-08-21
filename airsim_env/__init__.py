import gym
from gym.envs.registration import register

register(
    id='AirSim-v0',
    entry_point='airsim_env.envs:AirSim',
    max_episode_steps=150,
    reward_threshold=500,
)