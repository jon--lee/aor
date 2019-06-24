# Adaptive On-Policy Regularization

This code is based on work from UC Berkeley.


This repository contains code for evaluting the on-policy imitation learning algorithms on several modified OpenAI gym environments.
The purpose of this code is to illustrate that under certain conditions these algorithms can fail to converge to even locally optimal solutions. Adaptive on-policy regularization is implemented to help ensure convergence of the algorithms.

There are three algorithms tested: DAgger, Imitation Gradient, Multiple Imitation Gradient. Details can be found in the supporting paper.

Before running experiments, please install the modified version of the OpenAI gym.

[OpenAI gym.](https://github.com/jon--lee/gym)

The purpose of the modified environments to elicit the instability in the algorithms that would otherwise be obscured by some design choices in the default environments. For example, the modified cart-pole does not simply end after the angular deviation exceeds a certain amount. Similarly the cost is changed to measure average angular deviation rather than time-alive. This is because learning agents can "cheat" the time-alive reward using very suboptimal, unstable policies that manage to stay upright just barely.

Other modifications include a change for Hopper and Walker to record only reward from distance traveled rather than time-alive. time-alive gives the agent a way to artificially inflate reward, especially with the relatively biased supervised learning models used in this project.

A list of other packages and their versions are listed at the end.

# Running the Code
The easiest way to run the code is to simply run the shell scripts:

```
sh run_cartpole.sh sh
run_walker.sh
sh run_hopper.sh
```

These files will execute python programs to run the experiments. There are several options for the experiments. You can use the `force_mag` flag to change how much influence an action has on the dynamics. You can set the `--reg` flag to indicate whether or not to use AOPR.

These experiments will output data into a `data/` directory. You can generate the images used in the paper by running `analyze_cartpole_instantaneous.py`, `analyze_walker_instantaneous.py`, or  `analyze_hopper_instantaneous.py`

Alternatively, you can write your own plotting code. These files require that all the experiments in the shell scripts are run.

# Packages

This is the result of running `pip freeze --local` in my virtual environment.

```
absl-py==0.2.2
appnope==0.1.0
astor==0.7.1
atari-py==0.1.1
backcall==0.1.0
-e git+https://github.com/openai/baselines.git@412460873340cee353819bfec505f45a1900b4a2#egg=baselines
bleach==1.5.0
certifi==2018.4.16
cffi==1.11.5
chardet==3.0.4
click==6.7
cloudpickle==0.5.3
cycler==0.10.0
Cython==0.28.3
decorator==4.3.0
dill==0.2.8.2
future==0.16.0
gast==0.2.0
glfw==1.6.0
grpcio==1.13.0
-e git+https://github.com/jon--lee/gym.git@85fba2a1c20d202cfb6ca2446fc85c282cfcef29#egg=gym
html5lib==0.9999999
idna==2.7
imageio==2.3.0
ipython==6.4.0
ipython-genutils==0.2.0
jedi==0.12.1
joblib==0.12.0
kiwisolver==1.0.1
llvmlite==0.24.0
Markdown==2.6.11
matplotlib==2.2.2
mpi4py==3.0.0
mujoco-py==1.50.1.56
numba==0.39.0
numpy==1.14.5
opencv-python==3.4.1.15
parso==0.3.0
pexpect==4.6.0
pickleshare==0.7.4
Pillow==5.2.0
progressbar2==3.38.0
prompt-toolkit==1.0.15
protobuf==3.6.0
ptyprocess==0.6.0
pycparser==2.18
pyglet==1.3.2
Pygments==2.2.0
PyOpenGL==3.1.0
pyparsing==2.2.0
python-dateutil==2.7.3
python-utils==2.3.0
pytz==2018.5
pyzmq==17.0.0
requests==2.19.1
scikit-learn==0.19.1
scipy==1.1.0
simplegeneric==0.8.1
six==1.11.0
tensorboard==1.8.0
tensorflow==1.8.0
termcolor==1.1.0
tqdm==4.23.4
traitlets==4.3.2
urllib3==1.23
wcwidth==0.1.7
Werkzeug==0.14.1
zmq==0.0.0
```
