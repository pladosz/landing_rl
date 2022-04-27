import math
import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import Categorical
from collections import namedtuple

from common.buffers import *
from common.utils import *
from common.rdpg import *
from common.evaluator import *

import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import display
import argparse
from gym import spaces

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

#pybullet imports
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.single_agent_rl.TakeoffAviary import TakeoffAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from PIL import Image



#class NormalizedActions(gym.ActionWrapper): # gym env wrapper
#    def _action(self, action):

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RDPG')
    # Set Environment
    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    parser.add_argument('--env', default='gym_gazebo2_px4:px4-landing-v1', type=str, help='open-ai gym environment')

    # Set network parameter
    parser.add_argument('--hidden_1', default=300, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden_2', default=64, type=int, help='input num of GRU layer')
    parser.add_argument('--hidden_3', default=64, type=int, help='output num of GRU layer')
    parser.add_argument('--n_layers', default=1, type=int, help='number of stack for hidden layer')
    parser.add_argument('--rate', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--prate', default=0.00001, type=float, help='policy net learning rate (only for DDPG)')
    parser.add_argument('--discount', default=0.95, type=float, help='Discount factor for next Q values')
    parser.add_argument('--init_w', default=0.003, type=float, help='Initial network weight')
    parser.add_argument('--tau', default=0.0001, type=float, help='moving average for target network')
    parser.add_argument('--drop_prob', default=0.2, type=float, help='dropout_probability')

    # Set learning parameter
    parser.add_argument('--rbsize', default=20000, type=int, help='Memory size')
    parser.add_argument('--bsize', default=128, type=int, help='minibatch size')
    parser.add_argument('--blength', default=1, type=int, help='minibatch sequence length')
    parser.add_argument('--max_episodes', default=100000, type=int, help='Number of episodes')
    parser.add_argument('--max_episode_length', default=300, type=int, help='Number of steps for each episode')
    parser.add_argument('--validate_episodes', default=5, type=int, help='Number of episodes to perform validation')
    parser.add_argument('--validate_interval', default=1000, type=int, help='Validation episode interval')
    parser.add_argument('--epsilon_rate', default=0.1, type=float, help='linear decay of exploration policy')

    #etc
    parser.add_argument('--pause_time', default=0, type=float, help='Pause time for evaluation')
    parser.add_argument('--model_path', default='model/rdpg_pf_dir_vel_no_penalty_center_extreme_noisy_1004', type=str, help='Output root')
    parser.add_argument('--model_path_current', default='model/rdpg_pf_dir_vel_no_penalty_center_extreme_noisy_1004_current', type=str, help='Output root')
    parser.add_argument('--model_path_checkpoint', default='model/rdpg_landing_checkpoint.pt', type=str, help='checkpoint save file')
    parser.add_argument('--save_interval', default=100, type=int, help='how often checkpoint is saved')

    args = parser.parse_args()


    env = gym.make(args.env)
    action_space = env.action_space
    state_space  = env.observation_space
    batch_size = args.bsize  # each sample in batch is an episode for gru policy (normally it's timestep)
    update_itr = 1  # update iteration
    n_layers = args.n_layers
    replay_buffer_size = args.rbsize
    replay_buffer = ReplayBufferGRU(replay_buffer_size)
    args.model_path = get_output_folder(args.model_path, args.env)
    args.model_path_current = get_output_folder(args.model_path_current, args.env)
    torch.autograd.set_detect_anomaly(True)

    alg = RDPG(args, replay_buffer, state_space, action_space)
    evaluate = Evaluator(args)

    if args.mode == 'train':
        max_episodes  = args.max_episodes
        max_steps   = args.max_episode_length
        epsilon_steps = int(max_episodes*max_steps*args.epsilon_rate)
        
        state = env.reset()
        last_action = [0, 0, 0, 0]
        hidden_out = torch.zeros([n_layers, 1, args.hidden_3], dtype=torch.float).cuda()
        batch_length = 0

        for step in range(max_steps):
            hidden_in = hidden_out
            noise_level = max(epsilon_steps/epsilon_steps, 0)
            action, hidden_out = alg.policy_net.get_action(state, hidden_in, noise_level)
            print(action)
            next_state, reward, done, info = env.step(action)
            im = Image.fromarray(next_state.transpose(1, 2, 0), 'RGBA')
            im.save("test_images/drone_view_{0}.png".format(step))


            if batch_length==0:
                batch_state = []
                batch_action = []
                batch_last_action = []
                batch_reward = []
                batch_next_state = []
                batch_done = []
                ini_hidden_in = hidden_in
                ini_hidden_out = hidden_out

            batch_state.append(state)
            batch_action.append(action)
            batch_last_action.append(last_action)
            batch_reward.append(reward)
            batch_next_state.append(next_state)
            batch_done.append(done)

            state = next_state
            last_action = action
            batch_length += 1

            if batch_length == args.blength:
                batch_length = 0
                replay_buffer.push(ini_hidden_in, ini_hidden_out, batch_state, batch_action, batch_last_action, \
                                   batch_reward, batch_next_state, batch_done)

            if done:  # should not break for gru cases to make every episode with same length
                batch_length = 0
                break        

        exit()

