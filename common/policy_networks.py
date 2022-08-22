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
            self._action_dim = 2#action_space.n
        else:
            self._action_dim = 2#self._action_shape[0]
#        self.action_range = action_range
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

        # self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # self.gru1 = nn.GRU(1024, hidden_3, n_layers, dropout=drop_prob)
#        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        # self.linear3 = nn.Linear(hidden_3, self._action_dim) # output dim = dim of action
#################################################################################
        self.linear1 = nn.Linear(self._state_dim, hidden_1)
        #torch.nn.init.kaiming_uniform_(self.linear1.weight)
        self.linear2 = nn.Linear(hidden_1, hidden_2)
        #torch.nn.init.kaiming_uniform_(self.linear1.weight)
        # self.gru1 = nn.GRU(hidden_2, hidden_2, n_layers, dropout=drop_prob)
        self.linear3 = nn.Linear(hidden_2, self._action_dim)
        #torch.nn.init.xavier_uniform_(self.linear1.weight)
#################################################################################
        # weights initialization
        # self.linear3.weight.data.uniform_(-init_w, init_w)
        # self.linear3.bias.data.uniform_(-init_w, init_w)
    

    def forward(self, state, hidden_in):
        """ 
        state shape: (batch_size, sequence_length, state_dim)
        output shape: (batch_size, sequence_length, action_dim)
        for lstm needs to be permuted as: (sequence_length, batch_size, -1)
        """
#        print(hidden_in[0][0].size())
        #state = state.permute(1,0,2)
#        last_action = last_action.permute(1,0,2)
        #activation=F.relu
        # single branch
        # = torch.cat([state], -1)
        # x = state
        # x = F.relu(self.conv1(x))   # lstm_branch: sequential data
        # x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        # x = torch.flatten(x, start_dim = 1)
        # x = x.unsqueeze(0)
        # hidden only for initialization, later on hidden states are passed automatically for sequential data
        # x,  lstm_hidden = self.gru1(x, hidden_in)    # no activation after lstm
#        x = activation(self.linear2(x))
        # x = F.tanh(self.linear3(x))
        # x = x.permute(1,0,2)  # permute back
        # return x, lstm_hidden    # lstm_hidden is actually tuple: (hidden, cell)
        #############################################
        x = state
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        # x = torch.flatten(x, start_dim = 1)
        # x = x.unsqueeze(0)
        # x,  lstm_hidden = self.gru1(x, hidden_in)
        lstm_hidden = hidden_in
        # x = self.linear3(x)
        x = torch.tanh(self.linear3(x))
        x = torch.flatten(x, start_dim = 1)
        x = x.unsqueeze(0)
        return x, lstm_hidden


    def evaluate(self, state, hidden_in, noise_scale=0.0):
        '''
        evaluate action within GPU graph, for gradients flowing through it, noise_scale controllable
        '''
        # print("eval")
        normal = Normal(0, 1)
        action, hidden_out = self.forward(state, hidden_in)
        action = action.squeeze(0)
        # noise = noise_scale * normal.sample(action.shape).cuda()
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
        state = torch.FloatTensor(state).unsqueeze(0).cuda() # increase 2 dims to match with training data

#        last_action = torch.FloatTensor(last_action).unsqueeze(0).unsqueeze(0).cuda()
        normal = Normal(0,1)
        action, hidden_out = self.forward(state, hidden_in)
        # print("TRUE action : ", action)

        noise = noise_scale * normal.sample(action.shape).cuda()
        # action= torch.clip(action + noise, -1, 1)
        action= action + noise
        #print(action)
        # print("noise action", action)

        # while action[0][0][0] > 1: # Angle wrapping
        #     action[0][0][0] = action[0][0][0] - 2
        # while action[0][0][0] < -1:
        #     action[0][0][0] = action[0][0][0] + 2

        # while action[0][0][1] > 1: # Angle wrapping
        #     action[0][0][0] = action[0][0][0] - 2
        # while action[0][0][1] < -1:
        #     action[0][0][0] = action[0][0][0] + 2

        # while action[0][0][1] < 0:
        #     action[0][0][1] = 0
        #print("wrapped action", action)


        return action.detach().cpu().numpy()[0][0], hidden_out

    def sample_action(self):
        normal = Normal(0, 1)
        random_action=self.action_range*normal.sample( (self._action_dim,) )

        return random_action.cpu().numpy()
