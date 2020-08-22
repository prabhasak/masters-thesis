Masters-Thesis: Learning from demonstrations
==========================
**Codebase:** Benchmark reinforcement learning (RL) and imitation Learning (GAIL) algorithms from [Stable Baselines 2.10](https://stable-baselines.readthedocs.io/en/master/index.html) on [OpenAI Gym](https://gym.openai.com/) and [AirSim](https://microsoft.github.io/AirSim/) environments

**Framework, langauge, OS:** Tensorflow 1.14, Python 3.7, Windows 10

**Motivation:** The original Stable Baselines codebase includes a great set of features, 

**Idea**: pick {env, algo} pair -> train RL (optimal policy) -> generate expert data (optimal expert) -> train GAIL (policy)

**Thesis:** Autonomous UAV landing using human demonstrations alone (expert). Apply imitation learning methods on a custom environment built on [Microsoft AirSim 2.0](https://microsoft.github.io/AirSim/). Short video [here](https://www.youtube.com/watch?v=oj4y8GOq4gk&feature=youtu.be)

Prerequisites
-------------
The implementation uses [Stable Baselines 2.10](https://stable-baselines.readthedocs.io/en/master/guide/install.html). Note: You will need to install the OpenMPI version to use ``GAIL``, ``TRPO`` algorithms. I have inlcuded 'utils' from [here](https://github.com/araffin/rl-baselines-zoo) to save hyperparameters into a dictionary

AirSim: Some resources to [generate](https://microsoft.github.io/AirSim/build_windows/) custom binary files, modify [settings](https://microsoft.github.io/AirSim/settings/). Binaries for my thesis available [here](https://drive.google.com/drive/folders/1PFYkOlqb0DLcVoSsaSNGZVJif1VGeGuK?usp=sharing)

```
# create virtual environment (optional)
conda create -n myenv python==3.7
conda activate myenv

git clone https://github.com/prabhasak/masters-thesis.git

# install required libraries and modules (recommended)
pip install -r requirements.txt

# MPI needed for TRPO, GAIL
pip install stable-baselines[mpi]
```

Add your CustomEnv ID details to ``env_list, env_success, episode_len`` ([examples](https://github.com/openai/gym/blob/master/gym/envs/__init__.py)), and your custom RL/IL algorithm to ``algo_list`` (You can generate a CustomEnv ID by [registering](https://medium.com/@apoddar573/making-your-own-custom-environment-in-gym-c3b65ff8cdaa) your CustomEnv on Gym. You can use the ``"airsim_env"`` folder above as reference)

Usage
-------------
The codebase contains **[Tensorboard](https://stable-baselines.readthedocs.io/en/master/guide/tensorboard.html)** and **[Callback](https://stable-baselines.readthedocs.io/en/master/guide/callbacks.html)** features, which help monitor performance during training. You can enable them with ``-tb`` and ``-check, -eval``. The callbacks are for [saving](https://stable-baselines.readthedocs.io/en/master/guide/callbacks.html#checkpointcallback) the model periodically (useful for [continual learning](https://stable-baselines.readthedocs.io/en/master/guide/examples.html#continual-learning) and to resume training) and [evaluating](https://stable-baselines.readthedocs.io/en/master/guide/callbacks.html#evalcallback) the model periodically (saves the best model throughout training). You can choose to save and evaluate just the best model with ``-best``

**OpenAI Gym envs:** ``python imitation_learning_basic.py --seed 42 --env Pendulum-v0 --algo sac -rl -trl 1e5 -il -til 3e5 -best -check -eval -tb -params-RL learning_starts:1000 -params-IL lam:0.9 vf_iters:10`` (``--device`` assumes single-GPU machine. Exclude ``-rl`` if expert data is available)

Verify reproducibility: (i) 71/100 successful experts with mean(-145.37/-150.43), std(80.41/82.06), and (ii) 43/100 successful episodes on GAIL policy evaluation with mean(-227.61/-200.78), std(147.36/109.21)

**AirSim env:** ``python imitation_learning_basic.py --env AirSim-v0 --algo sac --exp-id 1 --seed 42 -rl -trl 4e5 -il -til 1e6 -params-IL gamma:0.995`` (run this after opening binaries/Windows/Blocks.exe or binaries/Linux/Blocks.sh)

Tuned hyperparameters (HPs) are available on [Baselines Zoo](https://github.com/araffin/rl-baselines-zoo/tree/master/hyperparams). Please read ``description.txt`` for more info on the sub-folders

Note: For [deterministic evaluation](https://github.com/hill-a/stable-baselines/issues/929#issuecomment-655319112) of expert data, add ``deterministic=True`` [here](https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/gail/dataset/record_expert.py#L120). To hide expert data info (keys, shape), you will have to comment [this](https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/gail/dataset/record_expert.py#L173)

Future Work
-------------
1. [Multiprocessing](https://stable-baselines.readthedocs.io/en/master/guide/vec_envs.html#subprocvecenv): to speed up training (observed 6x speedup on my PC)
2. [HP tuning](https://stable-baselines.readthedocs.io/en/master/guide/rl_zoo.html): to find the best set of hyperparameters for an (environment-algorithm) pair
3. [VecNormalize](https://stable-baselines.readthedocs.io/en/master/guide/vec_envs.html#vecnormalize): for normalizing data (useful for MuJoCo environments)
4. [Monitor](https://stable-baselines.readthedocs.io/en/master/common/monitor.html): for recording more internal state information
5a. Comparing consecutive runs of the experiment and picking the best model
5b. Loading tuned HPs and passing arguments to custom environments

I plan to release a separate repo for all of this once my reasearch is done!