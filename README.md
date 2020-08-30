Learning from demonstrations: An Imitation Learning benchmark for Stable Baselines 2.10
==========================
**Objective:** Benchmark reinforcement learning (RL) and imitation Learning ([GAIL](https://arxiv.org/pdf/1606.03476.pdf)) algorithms from [Stable Baselines 2.10](https://stable-baselines.readthedocs.io/en/master/index.html) on [OpenAI Gym](https://gym.openai.com/) and [AirSim](https://microsoft.github.io/AirSim/) environments. To be more specific, the goal of this codebase is to:
1. Train a GAIL model to <ins>imitate</ins> expert demonstrations <ins>generated</ins> from a trained RL model
2. Integrate several cool features provided by Stable Baselines (to the best of my knowledge, uncharted territory!)

**Idea**: Pick your favourite [task, RL algo] pair -> **train RL** -> **rollout expert data** -> **train GAIL** -> **verify imitation**

**Framework, langauge, OS:** Tensorflow 1.14, Python 3.7, Windows 10

**Thesis problem statement:** Imitate Autonomous UAV maneuver and landing purely from human demonstrations. We train GAIL on a custom environment built on [Microsoft AirSim 2.0](https://microsoft.github.io/AirSim/). Short video [here](https://www.youtube.com/watch?v=oj4y8GOq4gk&feature=youtu.be)

Prerequisites
-------------
The implementation uses [Stable Baselines 2.10](https://stable-baselines.readthedocs.io/en/master/guide/install.html). Inlcuded 'utils.py' from [here](https://github.com/araffin/rl-baselines-zoo) to save hyperparameters as a Dict object

```
# create virtual environment (optional)
conda create -n myenv python=3.7
conda activate myenv

git clone https://github.com/prabhasak/masters-thesis.git
cd masters-thesis
pip install -r requirements.txt # recommended
pip install stable-baselines[mpi] # MPI needed for TRPO, GAIL
```

**For CustomEnvs and CustomAlgos:** [Register](https://medium.com/@apoddar573/making-your-own-custom-environment-in-gym-c3b65ff8cdaa) your CustomEnv on Gym ([examples](https://github.com/openai/gym/blob/master/gym/envs/__init__.py)), and add your custom env and/or algorithm details to the code. You can use the ``"airsim_env"`` folder for reference

**For AirSim:** Some resources to [generate](https://microsoft.github.io/AirSim/build_windows/) custom binary files, modify [settings](https://microsoft.github.io/AirSim/settings/). Binaries for my thesis available [here](https://drive.google.com/drive/folders/1PFYkOlqb0DLcVoSsaSNGZVJif1VGeGuK?usp=sharing). You will have to run them _before_ running the code


Usage
-------------
1. **Train RL and GAIL:**

``python train.py --seed 42 --env Pendulum-v0 --algo sac -rl -trl 1e5 -il -til 3e5 -best -check -eval -tb -params-RL learning_starts:1000 -params-IL lam:0.9 vf_iters:10``

<!-- Verify reproducibility: (i) 70/100 successful experts with (mean, std) = (-152.93, 84.02) or (-149.43, 79.70), and 
ii) 54/100 or 43/100 successful episodes on GAIL policy evaluation with (mean, std) = (-193.65, 105.68) or (-216.51, 132.16) -->

Exclude ``-rl`` if expert data is available. For [deterministic evaluation](https://github.com/hill-a/stable-baselines/issues/929#issuecomment-655319112) of expert data, add ``deterministic=True`` [here](https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/gail/dataset/record_expert.py#L120). Tuned hyperparameters (HPs) are available on [Baselines Zoo](https://github.com/araffin/rl-baselines-zoo/tree/master/hyperparams). Please read ``description.txt`` for info on sub-folders

2. **Check expert data:** ``python expert_data_view.py --seed 42 --env Pendulum-v0 --algo sac --episodic``\
If ``--episodic``, use 'c' to go through each episode, and 'q' to stop the program

3. **Render expert data:** ``python expert_data_render.py --seed 42 --env My-Pendulum-v0 --algo sac --render``\
For envs in "custom_env" folder. If ``--episodic``, use 'c' to go through each episode, and 'q' to stop the program

4. **Evaluate, render model:** ``python model_render.py --seed 42 --env Pendulum-v0 --algo sac --mode rl -policy``\
Verify optimality of trained RL model and imitation accuracy of trained GAIL model. Include ``--test`` to render

<!-- To hide expert data info (keys, shape), you will have to comment [this](https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/gail/dataset/record_expert.py#L173) out -->

Features
-------------
The codebase contains **[Tensorboard](https://stable-baselines.readthedocs.io/en/master/guide/tensorboard.html)** and **[Callback](https://stable-baselines.readthedocs.io/en/master/guide/callbacks.html)** features, which help monitor performance during training. You can enable them with ``-tb`` and ``-check,-eval`` respectively. Usage: ``tensorboard --logdir "/your/file/path"``. Callbacks for:
1. [Saving](https://stable-baselines.readthedocs.io/en/master/guide/callbacks.html#checkpointcallback) the model periodically (useful for [continual learning](https://stable-baselines.readthedocs.io/en/master/guide/examples.html#continual-learning) and to resume training)
2. [Evaluating](https://stable-baselines.readthedocs.io/en/master/guide/callbacks.html#evalcallback) the model periodically and saves the best model throughout training (you can choose to save and evaluate just the best model found throughout training with ``-best``)

Future Work
-------------
1. [Multiprocessing](https://stable-baselines.readthedocs.io/en/master/guide/vec_envs.html#subprocvecenv): speed up training (observed 6x speedup for CartPole-v0 on my CPU with 12 threads)
2. [HP tuning](https://stable-baselines.readthedocs.io/en/master/guide/rl_zoo.html): find the best set of hyperparameters for an [env, algo] pair
3. [VecNormalize](https://stable-baselines.readthedocs.io/en/master/guide/vec_envs.html#vecnormalize): normalize env observation, action spaces (useful for MuJoCo environments)
4. [Monitor](https://stable-baselines.readthedocs.io/en/master/common/monitor.html): record internal state information during training (episode length, rewards)
5. (i) Comparing consecutive runs of the experiment, and (ii) passing arguments, HPs to custom environments

This is a work in progress (available [here](https://github.com/prabhasak/reproducibility)), but I hope to release clean code once my reasearch is done!
