Masters-Thesis: Learning from demonstrations
==========================
**Codebase:** Benchmark reinforcement learning (RL) and imitation Learning (GAIL) algorithms from [Stable Baselines 2.10](https://stable-baselines.readthedocs.io/en/master/index.html) on [OpenAI Gym](https://gym.openai.com/) and [AirSim](https://microsoft.github.io/AirSim/) environments

**Framework, langauge, OS:** Tensorflow 1.14, Python 3.7, Windows 10

**Motivation:** This is a humble attempt at combining all or most of the cool features provided by Stable Baselines! (not aware of any other efforts in this direction)

**Idea**: pick {env, algo} pair -> train RL (optimal policy) -> generate expert data (optimal expert) -> train GAIL (policy)

**Thesis:** Autonomous UAV landing using human demonstrations alone (expert). Apply imitation learning methods on a custom environment built on [Microsoft AirSim 2.0](https://microsoft.github.io/AirSim/). Short video [here](https://www.youtube.com/watch?v=oj4y8GOq4gk&feature=youtu.be)

Prerequisites
-------------
The implementation uses [Stable Baselines 2.10](https://stable-baselines.readthedocs.io/en/master/guide/install.html). I inlcuded 'utils' from [here](https://github.com/araffin/rl-baselines-zoo) to save hyperparameters into a dictionary

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

[Register](https://medium.com/@apoddar573/making-your-own-custom-environment-in-gym-c3b65ff8cdaa) your CustomEnv on Gym ([examples](https://github.com/openai/gym/blob/master/gym/envs/__init__.py)), and add your custom env, RL/IL algorithm details to the code You can use the ``"airsim_env"`` folder for reference

AirSim: Some resources to [generate](https://microsoft.github.io/AirSim/build_windows/) custom binary files, modify [settings](https://microsoft.github.io/AirSim/settings/). Binaries for my thesis available [here](https://drive.google.com/drive/folders/1PFYkOlqb0DLcVoSsaSNGZVJif1VGeGuK?usp=sharing). You will have to run them _before_ running the code.


Usage
-------------
The codebase contains **[Tensorboard](https://stable-baselines.readthedocs.io/en/master/guide/tensorboard.html)** and **[Callback](https://stable-baselines.readthedocs.io/en/master/guide/callbacks.html)** features, which help monitor performance during training. You can enable them with ``-tb`` and ``-check, -eval`` respectively. Callbacks are for:
1. [Saving](https://stable-baselines.readthedocs.io/en/master/guide/callbacks.html#checkpointcallback) the model periodically (useful for [continual learning](https://stable-baselines.readthedocs.io/en/master/guide/examples.html#continual-learning) and to resume training)
2. [Evaluating](https://stable-baselines.readthedocs.io/en/master/guide/callbacks.html#evalcallback) the model periodically and saves the best model throughout training (you can choose to save and evaluate just the best model with ``-best``)

**OpenAI Gym envs:** ``python imitation_learning_basic.py --seed 42 --env Pendulum-v0 --algo sac -rl -trl 1e5 -il -til 3e5 -best -check -eval -tb -params-RL learning_starts:1000 -params-IL lam:0.9 vf_iters:10`` (Exclude ``-rl`` if expert data is available. Also, ``--device`` assumes single-GPU machine)

**Verify reproducibility:** (i) 71/100 successful experts with mean(-145.37/-150.43), std(80.41/82.06), and 
ii) 43/100 successful episodes on GAIL policy evaluation with mean(-227.61/-200.78), std(147.36/109.21)

Tuned hyperparameters (HPs) are available on [Baselines Zoo](https://github.com/araffin/rl-baselines-zoo/tree/master/hyperparams). Please read ``description.txt`` for info on sub-folders

Note: For [deterministic evaluation](https://github.com/hill-a/stable-baselines/issues/929#issuecomment-655319112) of expert data, add ``deterministic=True`` [here](https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/gail/dataset/record_expert.py#L120)
<!-- To hide expert data info (keys, shape), you will have to comment [this](https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/gail/dataset/record_expert.py#L173) out -->

Future Work
-------------
1. [Multiprocessing](https://stable-baselines.readthedocs.io/en/master/guide/vec_envs.html#subprocvecenv): to speed up training (observed 6x speedup on my PC)
2. [HP tuning](https://stable-baselines.readthedocs.io/en/master/guide/rl_zoo.html): to find the best set of hyperparameters for an (environment-algorithm) pair
3. [VecNormalize](https://stable-baselines.readthedocs.io/en/master/guide/vec_envs.html#vecnormalize): for normalizing data (useful for MuJoCo environments)
4. [Monitor](https://stable-baselines.readthedocs.io/en/master/common/monitor.html): for recording more internal state information
5. (i) Comparing consecutive runs of the experiment and picking the best model (ii) Loading tuned HPs and passing arguments to custom environments

I plan to release a separate repo for all of this once my reasearch is done!