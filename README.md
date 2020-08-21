Masters-Thesis
==========================
**Codebase:** Basic implementation of Imitation Learning (GAIL) using algorithms from [Stable Baselines 2.10](https://stable-baselines.readthedocs.io/en/master/guide/install.html)

**Idea**: Pick {env, algo} -> Train RL (optimal policy) -> Generate expert data (optimal expert) -> Train GAIL (hopefully, optimal policy)

**Thesis:** Learning autonomous UAV landing from human demonstrations alone. Apply imitation learning methods on a custom environment built on [Microsoft AirSim 2.0](https://microsoft.github.io/AirSim/)

**Idea:** Human demonstrations (optimal expert) -> Train GAIL (hopefully, optimal policy)

Prerequisites
-------------

```
  conda create -n myenv python==3.7
  conda activate myenv
  git clone https://github.com/prabhasak/masters-thesis.git
  pip install -r requirements.txt
```

For AirSim: Some resources to [generate](https://microsoft.github.io/AirSim/build_windows/) your own binary files and modify [settings](https://microsoft.github.io/AirSim/settings/)

Usage
-------------

**OpenAI Gym envs:** ``python imitation_learning_basic.py --env Pendulum-v0 --algo sac --seed 42 -rl -trl 3e5 -il -til 3e5 -params gamma:0.995 timesteps_per_batch: 2048``

**AirSim env:** ``python imitation_learning_basic.py --env AirSim-v0 --algo sac --exp-id 1 --seed 42 -rl -trl 4e5 -il -til 1e6 -params gamma:0.995 timesteps_per_batch: 2048`` (run this after opening binaries/Windows/Blocks.exe or binaries/Linux/Blocks.sh)


Please read ``description.txt`` for more info on the sub-folders
