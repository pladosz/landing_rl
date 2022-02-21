import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import math
from .initialize import *

class PolicyNetworkBase(nn.Module):
    """ Base network class for policy function """
    def __init__(self, state_space, action_space):
        super(PolicyNetworkBase, self).__init__()
        self._state_space = state_space
        self._state_shape = state_space.shape
        if len(self._state_shape) == 1:
            self._state_dim = self._state_shape[0]
        else:  # high-dim state
            pass  
        self._action_space = action_space
        self._action_shape = action_space.shape
        if len(self._action_shape) < 1:  # Discrete space
            self._action_dim = action_space.n
        else:
            self._action_dim = self._action_shape[0]
#        self.action_range = action_range

    def forward(self):
        pass
    
    def evaluate(self):
        pass 
    
    def get_action(self):
        pass

    def sample_action(self,):
        a=torch.FloatTensor(self._action_dim).uniform_(-1, 1)
        return a.numpy()

class DPG_PolicyNetworkGRU(PolicyNetworkBase):
    """
    Deterministic policy gradient network with LSTM structure.
    The network follows single-branch structure as in paper: 
    Memory-based control with recurrent neural networks
    """
    def __init__(self, state_space, action_space, hidden_1, hidden_2, hidden_3, n_layers, drop_prob, init_w=3e-3):
        super().__init__(state_space, action_space)
#        self.hidden_1 = hidden_1
#        self.hidden_2 = hidden_2
#        self.hidden_3 = hidden_3

        self.linear1 = nn.Linear(self._state_dim, hidden_1)
        self.linear2 = nn.Linear(hidden_1, hidden_2)
        self.gru1 = nn.GRU(hidden_2, hidden_3, n_layers, dropout=drop_prob)
#        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_3, self._action_dim) # output dim = dim of action

        # weights initialization
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
    

    def forward(self, state, hidden_in):
        """ 
        state shape: (batch_size, sequence_length, state_dim)
        output shape: (batch_size, sequence_length, action_dim)
        for lstm needs to be permuted as: (sequence_length, batch_size, -1)
        """
#        print(hidden_in[0][0].size())
        state = state.permute(1,0,2)
#        last_action = last_action.permute(1,0,2)
        activation=F.relu
        # single branch
        x = torch.cat([state], -1)
        x = activation(self.linear1(x))   # lstm_branch: sequential data
        x = activation(self.linear2(x))
        # hidden only for initialization, later on hidden states are passed automatically for sequential data
        x,  lstm_hidden = self.gru1(x, hidden_in)    # no activation after lstm
#        x = activation(self.linear2(x))
        x = F.tanh(self.linear3(x))
        x = x.permute(1,0,2)  # permute back

        return x, lstm_hidden    # lstm_hidden is actually tuple: (hidden, cell)

    def evaluate(self, state, hidden_in, noise_scale=0.0):
        '''
        evaluate action within GPU graph, for gradients flowing through it, noise_scale controllable
        '''
        normal = Normal(0, 1)
        action, hidden_out = self.forward(state, hidden_in)
        noise = noise_scale * normal.sample(action.shape).cuda()
#        print(action)
#        action = action+noise
#        while action > 1: # Angle wrapping
#            action = action - 2
#            print("action_wrapped: ", action)
#        while action < -1:
#            action = action + 2
#            print("action_wrapped: ", action)

        return action, hidden_out

    def get_action(self, state, hidden_in,  noise_scale=1.0):
        '''
        select action for sampling, no gradients flow, noisy action, return .cpu
        '''
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).cuda() # increase 2 dims to match with training data
#        last_action = torch.FloatTensor(last_action).unsqueeze(0).unsqueeze(0).cuda()
        normal = Normal(0, 1)
        action, hidden_out = self.forward(state, hidden_in)
        #print("true action: ", action)

        noise = noise_scale * normal.sample(action.shape).cuda()
        action= action + noise
        #print("noise action", action)

        while action[0][0][0] > 1: # Angle wrapping
            action[0][0][0] = action[0][0][0] - 2
        while action[0][0][0] < -1:
            action[0][0][0] = action[0][0][0] + 2

        while action[0][0][1] < 0:
            action[0][0][1] = 0;
        #print("wrapped action", action)


        return action.detach().cpu().numpy()[0][0], hidden_out

    def sample_action(self):
        normal = Normal(0, 1)
        random_action=self.action_range*normal.sample( (self._action_dim,) )

        return random_action.cpu().numpy()
