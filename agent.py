import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import namedtuple, deque

from network import *


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, device, seed=0):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


class DQNagent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, params, device, replay_buffer=None, seed=0):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            params (Param type): parameters for training
            device (PyTorch device): hardware assignmnet
            replay_buffer(obj): replay buffer
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.params = params
        self.seed = seed
        self.device = device

        # Q-Network
        self.qnetwork_local = None
        self.qnetwork_target = None
        self.optimizer = None

        # Replay memory
        if replay_buffer is None:
            self.memory = ReplayBuffer(action_size=action_size     , buffer_size=params.buffer_size,
                                       batch_size=params.batch_size, device=device)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def model_register(self, name, checkpoint=None):
        """Register models by the name.
        
        Params
        ======
            name (string): model name
            checkpoint (string): model path
        """
        if name == 'QNetwork':
            self.qnetwork_local = QNetwork(self.state_size, self.action_size, self.seed).to(self.device)
            self.qnetwork_target = QNetwork(self.state_size, self.action_size, self.seed).to(self.device)

            if checkpoint is not None:
                model = torch.load(checkpoint)
                self.qnetwork_local.load_state_dict(model)
        elif name == 'DeulingNetwork':
            self.qnetwork_local = DeulingNetwork(self.state_size, self.action_size, self.seed).to(self.device)
            self.qnetwork_target = DeulingNetwork(self.state_size, self.action_size, self.seed).to(self.device)

            if checkpoint is not None:
                model = torch.load(checkpoint)
                self.qnetwork_local.load_state_dict(model)
        else:
            raise NotImplementedError(f'Model:{name} not implemented')
        
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.params.lr)

    def step(self, state, action, reward, next_state, done):
        """Queue experience in reply memory and make train the model.
        
        Params
        ======
            state (float): current state
            action (int): action to next state
            reward (int): given reward by the action
            next_state (float): next state
            done (bool): if the episodic task done
        """
    
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.params.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.params.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.params.gamma)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # compute and minimize the loss
        expected_Qs = self.qnetwork_local(states).gather(1, actions)
        next_Qs = self.qnetwork_target(next_states).detach().max(1, keepdim=True)[0]
        
        target_Qs = rewards + gamma * next_Qs * (1 - dones)
        
        loss = F.mse_loss(expected_Qs, target_Qs)
        self.optimizer.zero_grad()    
        loss.backward()
        self.optimizer.step()

        # update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.params.tau)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class DDQNagent(DQNagent):
    """Only overwrite learn()"""
    def __init__ (self, state_size, action_size, params, device, replay_buffer=None, seed=0):
        super().__init__(state_size, action_size, params, device, replay_buffer, seed)
        self.params = params
        
    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        "*** Double DQN ***"
        expected_Qs = self.qnetwork_local(states).gather(1, actions)
        next_action = self.qnetwork_local(next_states).detach()
        next_action = next_action.argmax(1, keepdim=True)

        next_Qs = self.qnetwork_target(next_states).detach().gather(1, next_action)
        target_Qs = rewards + gamma * next_Qs * (1 - dones)
        
        loss = F.mse_loss(expected_Qs, target_Qs)
        self.optimizer.zero_grad()    
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.params.tau)