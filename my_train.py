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


max_step = 50

errors = []

state=env.reset()
action = [0, 0, 0, 0]

red = [5, 5]

for step in range(max_step):
    
    next_state, reward, done, info = env.step(action)
    im_array = next_state.transpose(1, 2, 0)
    


    # Find the most black point
    darkest = 500
    black = [5, 5]

    for x in range(0, np.shape(im_array)[1]):
        for y in range(9, np.shape(im_array)[0]-9):
            csum = im_array[y][x][0]/3+im_array[y][x][1]/3+im_array[y][x][2]/3
            if csum < darkest:
                darkest = csum
                black = np.array([y, x])
    

    
    # Find the first red point
    found = False
    for x in range(0, np.shape(im_array)[1]):
        for y in range(9, np.shape(im_array)[0]-9):
            if (im_array[y][x][0] > 150 and im_array[y][x][1] < 100):
                red = np.array([y, x])
                found = True
                break
            if found:
                break
        if found:
            break


    im_array[red[0]][red[1]] = [255, 255, 255, 255]
    im_array[black[0]][black[1]] = [255, 255, 255, 255]

    im_array[32][32] = [128, 0, 128, 255]
    im_array[15][20] = [128, 0, 128, 255]


    e = np.zeros(4)
    e[0:2] = black-np.array([32, 32])
    e[2:4] = red - np.array([15, 20])
    errors.append(sum(e*e)**0.5)

    
    

    im = Image.fromarray(im_array, 'RGBA')
    im.save("test_images/drone_view_{0}.png".format(step))



    lamb = 0.1
    Z = 1
    Lx1 = np.array([[-1/Z, 0, black[1]/Z, black[0]*black[1], -(1+black[1]**2), black[0]],
                   [0, -1/Z, black[0]/Z, 1+black[0]**2, -black[0]*black[1], -black[1]]])

    Lx2 = np.array([[-1/Z, 0, red[1]/Z, red[0]*red[1], -(1+red[1]**2), red[0]],
                   [0, -1/Z, red[0]/Z, 1+red[0]**2, -red[0]*red[1], -red[1]]])    

    
    L_ep1 = np.linalg.pinv(Lx1)
    L_ep2 = np.linalg.pinv(Lx2)

    L_ep = []
    L_ep[0:6] = np.transpose(L_ep1)
    L_ep[6:12] = np.transpose(L_ep2)
    L_ep = np.transpose(L_ep)

    action = -lamb * np.dot(L_ep, e)
    action = action[0:4]

    # ad_z = 1 - 1/(1+np.exp(-1 * sum(e*e)**0.5))
    action[2] = action[2]

    action[3] = (action[0]**2+action[1]**2+action[2]**2)**(1/2)
    print("ERROR: ", e)
    print("ACTIOOOON: ", action, "\n")


# plt.plot(np.arange(1, max_step+1), errors)
# plt.show()