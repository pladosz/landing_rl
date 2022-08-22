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
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(0.1)

    def forward(self):
        pass

class QNetworkBase(ValueNetworkBase):
    def __init__(self, state_space, action_space, activation ):
        super().__init__( state_space, activation)
        self._action_space = action_space
        self._action_shape = action_space.shape
        self._action_dim = 2#self._action_shape[0]

class QNetworkGRU(QNetworkBase):
    """
    Q network with LSTM structure.
    The network follows single-branch structure as in paper: 
    Memory-based control with recurrent neural networks
    """
    def __init__(self, state_space, action_space, hidden_1, hidden_2, hidden_3, n_layers, drop_prob, activation=F.relu, output_activation=None):
        super().__init__(state_space, action_space, activation)

        # self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # self.gru1 = nn.GRU(1024, hidden_3, n_layers, dropout=drop_prob)

        # self.linear3 = nn.Linear(hidden_3, self._action_dim) # output dim = dim of action

        # self.linear3.apply(linear_weights_init)
        self.linear1 = nn.Linear(self._state_dim + self._action_dim, hidden_1)
        #torch.nn.init.kaiming_uniform_(self.linear1.weight)
        self.linear2 = nn.Linear(hidden_1, hidden_2)
        #torch.nn.init.kaiming_uniform_(self.linear2.weight)
        #self.linear3 = nn.Linear(hidden_2, hidden_2)
        # self.gru1 = nn.GRU(hidden_2, hidden_3, n_layers, dropout=drop_prob)
        self.linear4 = nn.Linear(hidden_3, 1)
        #torch.nn.init.xavier_uniform_(self.linear2.weight)
        # weights initialization
        # self.linear4.apply(linear_weights_init)
        
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
        #state = state.permute(1,0,2)
        # print("val net@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@22")
        # print(state.shape)
        ############################################################33
        # state = state.permute(1,0,2)
        # action = action.permute(1,0,2)
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))   # lstm_branch: sequential data
        x = F.relu(self.linear2(x))
        # x = torch.flatten(x, start_dim = 1)
        # x = x.unsqueeze(0)
        # x,  lstm_hidden = self.gru1(x, hidden_in)    # no activation after lstm
        lstm_hidden = hidden_in
        x = self.linear4(x)
        x = torch.flatten(x, start_dim = 1)
        x = x.unsqueeze(0)
        x = x.permute(1,0,2)  # back to same axes as input    
        return x, lstm_hidden    # lstm_hidden is actually tuple: (hidden, cell)
