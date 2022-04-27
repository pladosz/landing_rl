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


env = 'landing-aviary-v0'
env = gym.make(env)


max_step = 3

state=env.reset()
action = [1, 1, 1, 2]

for step in range(max_step):
    
    next_state, reward, done, info = env.step(action)
    im_array = next_state.transpose(1, 2, 0)
    im = Image.fromarray(im_array, 'RGBA')

    darkest = 500
    dark = [5, 5]

    for x in range(0, np.shape(im_array)[1]):
        for y in range(9, np.shape(im_array)[0]-9):
            csum = im_array[y][x][0]/3+im_array[y][x][1]/3+im_array[y][x][2]/3
            if csum < darkest:
                darkest = csum
                dark = np.array([y, x])

    im_array[dark[0]][dark[1]] = [255, 255, 255, 255]
    im_array[32][32] = [128, 0, 128, 255]

    im.save("test_images/drone_view_{0}.png".format(step))

    lamb = 0.1
    Z = 1
    Lx = np.array([[-1/Z, 0, dark[1]/Z, dark[0]*dark[1], -(1+dark[1]**2), dark[0]],
                   [0, -1/Z, dark[0]/Z, 1+dark[0]**2, -dark[0]*dark[1], -dark[1]]])

    e = dark-np.array([32, 32])
    # L_ep = np.dot(np.linalg.inv(np.dot(Lx.transpose(), Lx)), Lx.transpose())
    
    L_ep = np.linalg.pinv(Lx)

    print("Lep: ", L_ep)
    print("e_tr", np.transpose(e))
    print("inverse proof", np.dot(Lx, L_ep))
    action = -lamb * np.dot(L_ep, np.transpose(e))
    action = action[0:4]
    action[3] = (action[0]**2+action[1]**2+action[2]**2)**(1/2)
    print("ERROR: ", e)
    print("ACIOOOON: ", action, "\n")
