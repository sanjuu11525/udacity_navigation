import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Deep Q Model."""

    def __init__(self, state_size, action_size, seed=0):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.model = nn.Sequential(nn.Linear(state_size, 64), nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                   nn.Linear(64        , 64), nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                   nn.Linear(64        , 64), nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                   nn.Linear(64        , action_size))

    def forward(self, state):
        return self.model(state)


class DeulingNetwork(nn.Module):
    """Deuling Q Model."""

    def __init__(self, state_size, action_size, seed=0):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(DeulingNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.model = nn.Sequential(nn.Linear(state_size, 64), nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                   nn.Linear(64        , 64), nn.LeakyReLU(negative_slope=0.1, inplace=True),
                                   nn.Linear(64        , 64), nn.LeakyReLU(negative_slope=0.1, inplace=True))
                                   
        self.advantage = nn.Sequential(nn.Linear(64 , action_size))
        self.v_s = nn.Sequential(nn.Linear(64 , 1))                       


    def forward(self, state):
        x = self.model(state)

        advantage = self.advantage(x)
        v_s = self.v_s(x)

        out = v_s + advantage - advantage.mean(dim=1, keepdim=True)
        return out