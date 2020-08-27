import gym
from gym.envs.registration import register

register(
    id='My-Pendulum-v0',
    entry_point='custom_env.envs:PendulumEnv',
)
register(
    id='My-CartPole-v1',
    entry_point='custom_env.envs:CartPoleEnv',
)
register(
    id='My-LunarLander-v2',
    entry_point='custom_env.envs:LunarLander',
)
register(
    id='My-LunarLanderContinuous-v2',
    entry_point='custom_env.envs:LunarLanderContinuous',
)
# register(
#     id='My-BipedalWalker-v2',
#     entry_point='custom_env.envs:BipedalWalker',
# )
# register(
#     id='My-HalfCheetah-v2',
#     entry_point='custom_env.envs:HalfCheetahEnv',
# )
# register(
#     id='My-Hopper-v2',
#     entry_point='custom_env.envs:HopperEnv',
# )
# register(
#     id='My-Ant-v2',
#     entry_point='custom_env.envs:AntEnv',
# )
register(
    id='My-AirSim-v0',
    entry_point='custom_env.envs:AirSim',
)