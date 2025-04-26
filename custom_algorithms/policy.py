# custom_algorithms/my_policy.py

import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=256):
        """
        Defines a neural network that maps observations to actions. 
        
        Args:
            obs_dim = # inputs
            act_dim = # outputs
            hidden_size = # neurons in hidden layers 
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        # Outputs the mean of a normal distribution over the action space.
        self.mean = nn.Linear(hidden_size, act_dim)
        # Learnable log std for normal distribution. 
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):
        """ Returns a distribution over the action space, given
        an observation. The neural network returns the mean 
        and standard deviation of the distribution. """
        x = self.net(obs)
        mean = self.mean(x)
        std = torch.exp(self.log_std)  # Ensure std is positive
        dist = torch.distributions.Normal(mean, std)
        return dist
