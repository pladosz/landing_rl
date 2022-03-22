import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from .initialize import *

class ValueNetworkBase(nn.Module):
    """ Base network class for value function approximation """
    def __init__(self, state_space, activation):
        super(ValueNetworkBase, self).__init__()
        self._state_space = state_space
        self._state_shape = state_space.shape
        if len(self._state_shape) == 1:
            self._state_dim = self._state_shape[0]
        else:  # high-dim state
            pass  

        self.activation = activation

    def forward(self):
        pass

class QNetworkBase(ValueNetworkBase):
    def __init__(self, state_space, action_space, activation ):
        super().__init__( state_space, activation)
        self._action_space = action_space
        self._action_shape = action_space.shape
        self._action_dim = self._action_shape[0]

class QNetworkGRU(QNetworkBase):
    """
    Q network with LSTM structure.
    The network follows single-branch structure as in paper: 
    Memory-based control with recurrent neural networks
    """
    def __init__(self, state_space, action_space, hidden_1, hidden_2, hidden_3, n_layers, drop_prob, activation=F.relu, output_activation=None):
        super().__init__(state_space, action_space, activation)
#        self.hidden_1 = hidden_1
        self.linear1 = nn.Linear(self._state_dim+self._action_dim, hidden_1)
        self.linear2 = nn.Linear(hidden_1, hidden_2)
        self.gru1 = nn.GRU(hidden_2, hidden_3, n_layers, dropout=drop_prob)
#        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_3, 1)
        # weights initialization
        self.linear3.apply(linear_weights_init)
        
    def forward(self, state, action, hidden_in):
        """ 
        state shape: (batch_size, sequence_length, state_dim)
        output shape: (batch_size, sequence_length, 1)
        for lstm needs to be permuted as: (sequence_length, batch_size, state_dim)
        """
#        print(state.size())
#        print(state)
#        state.unsqueeze_(-1)
#        state = state.expand(32,1,9)
#        print(state.size())
#        print(state)
        state = state.permute(1,0,2)
        action = action.permute(1,0,2)
#        last_action = last_action.permute(1,0,2)
        # single branch
        x = torch.cat([state, action], -1) 
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        x, lstm_hidden = self.gru1(x, hidden_in)  # no activation after lstm
#        x = self.activation(self.linear2(x))
        x = self.linear3(x)
        x = x.permute(1,0,2)  # back to same axes as input    
        return x, lstm_hidden    # lstm_hidden is actually tuple: (hidden, cell)
