from audioop import add
import gym
import os
import numpy as np
from math import cos, sin, sqrt
import matplotlib.pyplot as plt
from gym_pybullet_drones.envs.single_agent_rl.TakeoffAviary import TakeoffAviary
from scipy.spatial.transform import Rotation
from PIL import Image
import pybullet
import time
import copy
import xml.etree.ElementTree as ET

# mycar = ET.parse('/home/user/landing/g_vehicle/car_v3.urdf')
# myroot = mycar.getroot()

# for prices in myroot.iter('name'):
#     print(prices.text)

# print('\n\n\n\n\n\n\n\n\n\n')

env = gym.make('landing-aviary-v0', gui = True)
# pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_SHADOWS,0)

os.system('rm -rf /home/user/landing/landing_rl/test_images')
os.system('mkdir /home/user/landing/landing_rl/test_images')
seed = 123456
max_step = 400
env.seed(seed+20)
state=env.reset()
action = [0, 0, 0]

zs = []
zsm=[]
e_x = []
e_xr = []
vx_rec = []
vx_act = []

e_y = []
e_yr = []
vy_rec = []
vy_act = []
lamb = 0.10


def add_debug_line():
    pybullet.addUserDebugLine(xyz, xyz + action, lifeTime=0.1,lineColorRGB = [150,100,0])
    pybullet.addUserDebugLine(xyz, xyz + e, lifeTime=0.1,lineColorRGB = [0,0,0])
    pybullet.addUserDebugText('{}'.format(step), (1,1,0), textColorRGB = (0,0,0),lifeTime  =0.1)
    pybullet.addUserDebugLine(xyz_past, xyz, lifeTime=0 ,lineColorRGB = [150,100,0], lineWidth = 2)
    pybullet.addUserDebugLine(xyz_GV_past, xyz_GV, lifeTime=0 ,lineColorRGB = [255,0,0], lineWidth = 2)


# Initial position of the vehicle
p_vhc_0 = np.array([env.pos[0,0], env.pos[0,1], 0.281])
drone_height_init = env.pos[0,2]
Z = drone_height_init-p_vhc_0[2]

I = 0
D = 0

for step in range(max_step):
    # Take the step and extract data
    img, reward, done, info = env.step(action)
    res = np.array(img.shape[0:2])

    xyz = info["drone_state"][0]
    rpy = info["drone_state"][1]
    quat = info["drone_state"][2]

    R = Rotation.from_quat(quat).as_matrix()

    xyz_GV = info["GV_state"]

    if step == 0:
        xyz_past = xyz
        xyz_GV_past = copy.deepcopy(xyz_GV)

    if done == True:
        break
    

    # Get matrix of red pixels
    red_map = np.zeros((res[0],res[1]))
    
    for x in range(0, res[1]):
        for y in range(0, res[0]):
            red_map[y][x] = ((img[y][x][0] > 100 and img[y][x][1] < 50 and img[y][x][2] < 50) or 
                             (img[y][x][0] < 50  and img[y][x][1] < 50 and img[y][x][2] < 50))



    # Estimate pad coordinate
    area_temp = np.sum(red_map)

    red_mul = np.dot(np.vstack((red_map, red_map.T)), np.arange(0, res[0]))/area_temp
    pad_center = np.array([np.sum(red_mul[res[0]:2*res[0]]), np.sum(red_mul[0:res[0]])])

    e_pixel = (res-1)/2-pad_center



    # Estimate z
    if step == 0:
        area0 = area_temp
        area = area0

        # Get focal lengths
        f_x = abs(Z * e_pixel[0] / p_vhc_0[0])
        f_y = abs(Z * e_pixel[1] / p_vhc_0[1])

        f_x = 40
        f_y = 40


    if area_temp > area:
        area = area_temp
        Z = env.pos[0,2] -0.281

    if area_temp <= area0*0.5:
        action = [0,0,1]
        continue

    # if area_temp > res[0] * res[1] * 0.8:
    #     continue


    # Get error in uav and world frames, in meters
    x = e_pixel[0] * Z / f_x
    y = e_pixel[1] * Z / f_y
    z = (-Z - R[2][0]*x-R[2][1]*y)/R[2][2]
    e_uav = np.array([x, y, z])


    # e = np.dot(R, e_uav)
    e = e_uav


    # Ascending velocity damping factor
    ad_z = 1 - 1/(1+np.exp(-0.01 * np.linalg.norm(e[0:2])))

    # if Z < 5:
    #     lamb = 0.15
    # if Z < 3:
    #     lamb = 0.2
    # if Z < 1:
    #     lamb = 0.4

    # vz = -0.35
    # vz = 0.15 * e[2] - 0.15 * ad_z
    vz = 0.1 * e[2]

    vx = lamb * e[0] - vz * e[0]/Z

    # vx = lamb*e[0] - vz * e[0]/Z + 0.05
    vy = lamb*e[1] - vz * e[1]/Z

    if np.isnan(vx):
        vx = 0 
    if np.isnan(vy):
        vy = 0

    action = np.array([vx, vy, vz])
    zs.append(Z)
    zsm.append(xyz[2]-0.281)
    e_x.append(e[0])
    e_xr.append(xyz_GV[0] - xyz[0])
    vx_rec.append(env.vel[0][0])
    vx_act.append(action[0])
    e_y.append(e[1])
    e_yr.append(xyz_GV[1] - xyz[1])
    vy_rec.append(env.vel[0][1])
    vy_act.append(action[1])


    xyz_past = copy.deepcopy(xyz)
    xyz_GV_past = copy.deepcopy(xyz_GV)

    # Mark the found points
    img[round(pad_center[0])][round(pad_center[1])] = [255, 255, 255, 255]
    img[round(res[0]/2-1)][round(res[1]/2-1)] = [128, 0, 128, 255]

    im = Image.fromarray(img, 'RGBA')
    im.save("test_images/drone_view_{0}.png".format(step))
    print("Step:", step, " | Z:", "%.3f" % Z, "| Action:", "%.7f" %action[0],"%.7f" %action[1],"%.7f" %action[2], "\n\n")
    
    if pybullet.getContactPoints(bodyA = 1) != ():
        break

fig1 = plt.figure(figsize = (8, 8))
plt.subplot(3, 1, 1)
plt.plot(zs, label='Z')
plt.plot(zsm, label='Real Z')
plt.grid()
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(e_x, label='e_x est')
plt.plot(e_xr, label='e_x real')
plt.plot(vx_rec, '--k' , label='vx sensor')
plt.plot(vx_act, '--k' , label='vx action')
plt.grid()
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(e_y, label='e_y est')
plt.plot(e_yr, label='e_y real')
plt.plot(vy_rec, '--k' , label='vy sensor')
plt.plot(vy_act, '--k' , label='vy action')
plt.grid()
plt.legend()

plt.show()