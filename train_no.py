

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RDPG')
    # Set Environment
    parser.add_argument('--env', default='gym_gazebo2_px4:px4-landing-v1', type=str, help='open-ai gym environment')


    parser.add_argument('--model_path', default='model/rdpg_pf_dir_vel_no_penalty_center_extreme_noisy_1004', type=str, help='Output root')
    parser.add_argument('--model_path_current', default='model/rdpg_pf_dir_vel_no_penalty_center_extreme_noisy_1004_current', type=str, help='Output root')
    parser.add_argument('--model_path_checkpoint', default='model/rdpg_landing_checkpoint.pt', type=str, help='checkpoint save file')
    parser.add_argument('--save_interval', default=100, type=int, help='how often checkpoint is saved')
    parser.add_argument('--load_checkpoint', default = '', type = str, help= 'string for loading data')


    args = parser.parse_args()


    env = gym.make(args.env)
    action_space = env.action_space
    state_space  = env.observation_space

    args.model_path = get_output_folder(args.model_path, args.env)
    args.model_path_current = get_output_folder(args.model_path_current, args.env)
    torch.autograd.set_detect_anomaly(True)

    #evaluate = Evaluator(args)