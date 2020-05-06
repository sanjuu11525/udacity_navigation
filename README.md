
# Udacity Navigation Project

## Introduction
This project is to solve the Udacity exercise, banana collection, with different DQN models. For this project, an agent has been trained to navigate a large square world.

### The Environment

The state space has 37 dimensions including the agent's velocity, and forward perception of objects. The agent has four discrete actions available

    0 - move forward.
    1 - move backward.
    2 - turn left.
    3 - turn right.

A reward of +1 is given for reaching a yellow banana, and a reward of -1 is given for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

## Getting Start

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)



The environment has two disctinct scenes. Refer to each specific page for setup, training and results.

## Installation

1. Clone the repository and initialize the submodules.

```
git clone https://github.com/sanjuu11525/udacity_navigation.git
cd udacity_navigation 
```

2. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) ,JupytorLab and  create a new conda env.

```
conda create -n ENV_NAME python=3.6
conda activate ENV_NAME
pip install jupyterlab
```

3. Install the project requirements.

```
pip install -r requirements.txt
```
## Running the Code

1. This repository is for Udacity Navigation project of Reinforcement Leaning Nanodegree. Some implementation is publicly avaialble.

2. Training the agent with implemented agents by ```train_dqn.ipynb```.

3. Evaluate pre-trained model by runnung ```python eval.py```. Please turn your jupytor down when evaluating training result. Otherwise, resource conficts.

4. Checkpoints are in ```./checkpoint```.

5. Vanilla DQN, Double DQN, and Deuling DQN are available.


## Reference

[1]https://github.com/udacity/deep-reinforcement-learning#dependencies
