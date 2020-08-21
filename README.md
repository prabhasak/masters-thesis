Masters-Thesis
==========================
**Codebase:** Basic implementation of Imitation Learning (GAIL) using algorithms from [Stable Baselines 2.10](https://stable-baselines.readthedocs.io/en/master/index.html)

**Framework, langauge:** Tensorflow 1.14, Python 3.7

**Idea**: pick {env, algo} -> train RL (optimal policy) -> generate expert data (optimal expert) -> train GAIL (policy)

**Thesis:** Autonomous UAV landing using human demonstrations alone (expert). Apply imitation learning methods on a custom environment built on [Microsoft AirSim 2.0](https://microsoft.github.io/AirSim/). Short video [here](https://www.youtube.com/watch?v=oj4y8GOq4gk&feature=youtu.be)

Prerequisites
-------------
The implementation uses [Stable Baselines 2.10](https://stable-baselines.readthedocs.io/en/master/guide/install.html). Note: You will need to install the OpenMPI version to use ``GAIL``, ``TRPO`` algorithms. Please include the 'utils' folder from [here](https://github.com/araffin/rl-baselines-zoo) in your cloned Stable Baselines repo

AirSim: Some resources to [generate](https://microsoft.github.io/AirSim/build_windows/) your own binary files and modify [settings](https://microsoft.github.io/AirSim/settings/). Binaries for my Thesis can be found [here](https://drive.google.com/drive/folders/1PFYkOlqb0DLcVoSsaSNGZVJif1VGeGuK?usp=sharing)

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

Usage
-------------
**OpenAI Gym envs:** ``python imitation_learning_basic.py --env Pendulum-v0 --algo sac --seed 42 -rl -trl 3e5 -il -til 3e5 -params-RL gamma:0.995 timesteps_per_batch: 2048`` (exclude ``-rl`` if expert data is available)

**AirSim env:** ``python imitation_learning_basic.py --env AirSim-v0 --algo sac --exp-id 1 --seed 42 -rl -trl 4e5 -il -til 1e6 -params-IL gamma:0.995`` (run this after opening binaries/Windows/Blocks.exe or binaries/Linux/Blocks.sh)


Please read ``description.txt`` for more info on the sub-folders
