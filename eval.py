from unityagents import UnityEnvironment
import numpy as np
import torch
import os

from agent import *

class Params:
    """Set up configuration here."""
    def __init__(self):
        self.__dict__.update(**{
            'buffer_size' : int(1e5),  # replay buffer size
            'batch_size' : 64,         # minibatch size
            'gamma' : 0.99,            # discount factor
            'tau' : 1e-3,              # for soft update of target parameters
            'lr' : 5e-4,               # learning rate 
            'update_every' : 4,        # how often to update the network
})

if __name__ == '__main__':

    # env setup
    env_file_name = os.path.abspath("Banana_Linux/Banana.x86_64")
    env = UnityEnvironment(file_name=env_file_name, no_graphics=False)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents in the environment
    print('Number of agents:', len(env_info.agents))

    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)

    # examine the state space 
    state = env_info.vector_observations[0]
    print('States look like:', state)
    state_size = len(state)
    print('States have length:', state_size)

    device = torch.device("cpu")

    # vanilla DQN
    model_name = 'QNetwork'
    agent = DQNagent(state_size=state_size, action_size=action_size, params=Params(), device=device)
    # model path
    model_path = os.path.abspath('checkpoint/checkpoint_dqn.pth')
    agent.model_register(model_name, model_path)

    for i_episode in range(1, 10):
        env_info = env.reset(train_mode=False)[brain_name] # reset the environment
        state = env_info.vector_observations[0]            # get the current state
        score = 0
        while True:
            action = agent.act(state)
            env_info   = env.step(action)[brain_name]      # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward     = env_info.rewards[0]               # get the reward
            done       = env_info.local_done[0]            # see if episode has finished
            state = next_state
            score += reward
            if done:
                break
        
        print(f'episode: {i_episode}, score: {score}')
