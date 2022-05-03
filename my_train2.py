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


max_step = 500
step = 0
zs = []
ars = []

state=env.reset()
action = [0, 0, 0]

for step in range(max_step):
    step+=1
    
    next_state, reward, done, info = env.step(action)
    im_array = next_state.transpose(1, 2, 0)
    
    # Getting the map of red area
    red_ctr = np.zeros((64, 64))
    red_sum= [0, 0]
    for x in range(0, np.shape(im_array)[1]):
        for y in range(0, np.shape(im_array)[0]):
            if (im_array[y][x][0] >100 and im_array[y][x][1] < 50 and im_array[y][x][2] < 50):
                red_ctr[y][x] = 1
                red_sum[0] += y
                red_sum[1] += x

    if np.sum(red_ctr) == 0:
        red_ctr[32][32] = 1
    
    red_mean =[int(red_sum[0] / np.sum(red_ctr)), int(red_sum[1] / np.sum(red_ctr))]

    # Find the rightmost red point
    red_r = [32, 32]
    for x in range(np.shape(im_array)[1]-1, red_mean[1], -1):
        y = red_mean[0]
        if (im_array[y][x][0] > 150 and im_array[y][x][1] < 100):
            red_r = np.array([y, x])
            break
    
    # Find the leftmost red point
    red_l = [32, 32]
    for x in range(0, red_mean[1]):
        y = red_mean[0]
        if (im_array[y][x][0] > 150 and im_array[y][x][1] < 100):
            red_l = np.array([y, x])
            break

    # Find the uppermost red point
    red_u = [32, 32]
    for y in range(0, red_mean[0]):
        x = red_mean[1]
        if (im_array[y][x][0] > 150 and im_array[y][x][1] < 100):
            red_u = np.array([y, x])
            break
    
    # Find the bottommost red point
    red_b = [32, 32]
    for y in range(np.shape(im_array)[0]-1, red_mean[0], -1):
        x = red_mean[1]
        if (im_array[y][x][0] > 150 and im_array[y][x][1] < 100):
            red_b = np.array([y, x])
            break

    # Get the average radius
    r_r = red_r[1] - red_mean[1]
    r_l = red_mean[1] - red_l[1]
    r_u = red_mean[0] - red_u[0]
    r_b = red_b[0] - red_mean[0]
    r_ave = (r_r+r_l+r_u+r_b)/4

    # Set the desired postions

    des_m = [32, 32]
    des_r = [des_m[0], des_m[1]+int(r_ave)]
    des_l = [des_m[0], des_m[1]-int(r_ave)]
    des_u = [des_m[0]-int(r_ave), des_m[1]]
    des_b = [des_m[0]+int(r_ave), des_m[1]]

    # Checking the found points
    im_array[red_mean[0]][red_mean[1]] = [255, 255, 255, 255]
    im_array[red_r[0]][red_r[1]] = [255, 255, 255, 255]
    im_array[red_l[0]][red_l[1]] = [255, 255, 255, 255]
    im_array[red_u[0]][red_u[1]] = [255, 255, 255, 255]
    im_array[red_b[0]][red_b[1]] = [255, 255, 255, 255]

    im_array[des_m[0]][des_m[1]] = [128, 0, 128, 255]
    im_array[des_r[0]][des_r[1]] = [128, 0, 128, 255]
    im_array[des_l[0]][des_l[1]] = [128, 0, 128, 255]
    im_array[des_u[0]][des_u[1]] = [128, 0, 128, 255]
    im_array[des_b[0]][des_b[1]] = [128, 0, 128, 255]

    im = Image.fromarray(im_array, 'RGBA')
    im.save("test_images/drone_view_{0}.png".format(step))

    area = np.sum(red_ctr)
    area0 = 28

    Z = 0.80*(area0/area)**(1/2) - 0.143

    if Z < 0.03 or Z > 1:
        Z = 0.03
    
    zs.append(Z)
    ars.append(area)


    lamb = 0.001

    # Defining the errors
    e = np.zeros(10)
    e[0:2] = np.array(des_m) - red_mean
    e[2:4] = np.array(des_r) - red_r
    e[4:6] = np.array(des_l) - red_l
    e[6:8] = np.array(des_u) - red_u
    e[8:10] = np.array(des_b) - red_b

    Lx1 = np.array([[-1/Z, 0, red_mean[1]/Z, red_mean[0]*red_mean[1], -(1+red_mean[1]**2), red_mean[0]],
                   [0, -1/Z, red_mean[0]/Z, 1+red_mean[0]**2, -red_mean[0]*red_mean[1], -red_mean[1]]])

    Lx2 = np.array([[-1/Z, 0, red_r[1]/Z, red_r[0]*red_r[1], -(1+red_r[1]**2), red_r[0]],
                   [0, -1/Z, red_r[0]/Z, 1+red_r[0]**2, -red_r[0]*red_r[1], -red_r[1]]])    

    Lx3 = np.array([[-1/Z, 0, red_l[1]/Z, red_l[0]*red_l[1], -(1+red_l[1]**2), red_l[0]],
                   [0, -1/Z, red_l[0]/Z, 1+red_l[0]**2, -red_l[0]*red_l[1], -red_l[1]]])    

    Lx4 = np.array([[-1/Z, 0, red_u[1]/Z, red_u[0]*red_u[1], -(1+red_u[1]**2), red_u[0]],
                   [0, -1/Z, red_u[0]/Z, 1+red_u[0]**2, -red_u[0]*red_u[1], -red_u[1]]])    

    Lx5 = np.array([[-1/Z, 0, red_b[1]/Z, red_b[0]*red_b[1], -(1+red_b[1]**2), red_b[0]],
                   [0, -1/Z, red_b[0]/Z, 1+red_b[0]**2, -red_b[0]*red_b[1], -red_b[1]]])  


    L_ep = np.zeros((10, 6))
    L_ep[0:2][:] = np.transpose(np.transpose(Lx1))
    L_ep[2:4][:] = np.transpose(np.transpose(Lx2))
    L_ep[4:6][:] = np.transpose(np.transpose(Lx3))
    L_ep[6:8][:] = np.transpose(np.transpose(Lx4))
    L_ep[8:10][:] = np.transpose(np.transpose(Lx5))


    L_ep = np.linalg.pinv(L_ep)

    action = -lamb * np.dot(L_ep, e)[0:3]

    ad_z = 1 - 1/(1+np.exp(-0.001 * sum(e*e)**0.5))
    action[0] += 0.65*0
    action[2] = ad_z*(action[2] - 0.2)
    


    print("Step:", step, "| Area:", area, "| Z:", "%.3f" % Z, "| Action:", "%.7f" %action[0],"%.7f" %action[1],"%.7f" %action[2], "\n")
    print("error: ",e)
    print("r: ",red_r)
    print("l: ",red_l)
    print("u: ",red_u)
    print("b: ",red_b)

plt.figure()
plt.plot(zs)
plt.figure()
plt.plot(ars)
plt.show()