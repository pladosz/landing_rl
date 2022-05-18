import gym
import os
import numpy as np
from math import cos, sin, sqrt
import matplotlib.pyplot as plt
from gym_pybullet_drones.envs.single_agent_rl.TakeoffAviary import TakeoffAviary
from scipy.spatial.transform import Rotation
from PIL import Image
import pybullet

env = gym.make('landing-aviary-v0', gui = True)
pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_SHADOWS,0)

os.system('rm -rf /home/user/landing/landing_rl/test_images')
os.system('mkdir /home/user/landing/landing_rl/test_images')

max_step = 250

state=env.reset()
action = [0, 0, 0]

zs = []
zsm=[]
e_x = []
e_y = []

lamb = 1.5
# Initial position of the vehicle
p_vhc_0 = np.array([0.2, 0.2, -0.719])

Z = 0.719

for step in range(max_step):
    # Take the step and extract data
    img, reward, done, info = env.step(action)
    res = np.shape(img)[0:2]
    xyz = info[0]

    # Rotation Matrix from the world to drone frame
    R = Rotation.from_quat(info[2]).as_matrix()
    
    # Get the map of red area
    red_map = np.zeros((res[0],res[1]))
    for x in range(0, res[1]):
        for y in range(0, res[0]):
            red_map[y][x] = (img[y][x][0] >100 and img[y][x][1] < 50 and img[y][x][2] < 50)

    # Get Z and Area approximations
    area_temp = np.sum(red_map)

    if area_temp <= 10:
        action = [0,0,0.1]
        continue

    red_mul = np.dot(np.vstack((red_map, red_map.T)), np.arange(0, res[0]))/area_temp
    red_m = np.array([np.sum(red_mul[res[0]:2*res[0]]), np.sum(red_mul[0:res[0]])])

    e_ = np.array([(res[1]-1)/2 - red_m[0],       # in uav frame, pixels
                   (res[0]-1)/2 - red_m[1]])

    if step == 0:
        area0 = area_temp
        area = area0
        f_x = abs(Z * e_[0] / p_vhc_0[0])
        f_y = abs(Z * e_[1] / p_vhc_0[1])

    if area_temp > area:
        area = area_temp
        Z = 0.719*(area0/area)**(1/2)-0.05

    x = e_[0] * Z / f_x
    y = e_[1] * Z / f_y
    z = (-Z - R[2][0]*x-R[2][1]*y)/R[2][2]
    
    e_uav = np.array([x, y, z])                 # in uav frame, meters

    e = np.dot(R, e_uav)

    # Ascending velocity damping factor
    ad_z = 1 - 1/(1+np.exp(-0.01 * np.linalg.norm(e)))

    # vz = vz_0 * ad_z
    vz = 0.5 * e[2]
    vx = 0.1*lamb*e[0] - vz * e[0]/Z + 0.2
    vy = 0.1*lamb*e[1] - vz * e[1]/Z

    action = np.array([vx, vy, vz])

    zs.append(Z)
    zsm.append(xyz[2]-0.281)
    e_x.append(e[0])
    e_y.append(e[1])

    pybullet.addUserDebugLine(xyz, xyz + action, lifeTime=0.1,lineColorRGB = [150,100,0])
    pybullet.addUserDebugLine(xyz, xyz + e, lifeTime=0.1,lineColorRGB = [0,0,0])
    pybullet.addUserDebugText('{}'.format(step), (1,1,0), textColorRGB = (0,0,0),lifeTime  =0.1)

    # Mark the found points
    img[round(red_m[0])][round(red_m[1])] = [255, 255, 255, 255]
    img[round(res[0]/2-1)][round(res[1]/2-1)] = [128, 0, 128, 255]

    im = Image.fromarray(img, 'RGBA')
    im.save("test_images/drone_view_{0}.png".format(step))
    print("Step:", step, " | Z:", "%.3f" % Z, "| Action:", "%.7f" %action[0],"%.7f" %action[1],"%.7f" %action[2], "\n\n")

plt.subplot(2, 1, 1)
plt.plot(zs, label='Z')
plt.plot(zsm, label='Real Z')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(e_x, label='e_x')
plt.plot(e_y, label='e_y')
plt.legend()

plt.show()