import gym
import os
import numpy as np
from math import cos, sin, sqrt
import matplotlib.pyplot as plt
from gym_pybullet_drones.envs.single_agent_rl.TakeoffAviary import TakeoffAviary
from scipy.spatial.transform import Rotation
from PIL import Image
import pybullet

env = gym.make('landing-aviary-v0')

pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_SHADOWS,0)

os.system('rm -rf /home/user/landing/landing_rl/test_images')
os.system('mkdir /home/user/landing/landing_rl/test_images')

max_step = 250
step = 0
result_freq = 5

state=env.reset()
action = [0, 0, 0]

zs = []
zsm=[]
ars = []
ers = []
e_x = []
e_y = []
e_z = []

lamb = 1.5

# Initial position of the vehicle
p_vhc_0 = np.array([0.2, 0.2, -0.719])

# z velocity of the UAV
vz = -0.2

for step in range(max_step):
    step+=1
    
    # Take the step and extract data
    img, reward, done, info = env.step(action)
    res = np.shape(img)[0:2]
    xyz = info[0]
    rpy = info[1]
    quat = info[2]

    # Rotation Matrix from the world to drone frame
    R = Rotation.from_quat(quat).as_matrix()
    
    # Get the map of red area
    red_map = np.zeros((res[0],res[1]))
    for x in range(0, res[1]):
        for y in range(0, res[0]):
            red_map[y][x] = (img[y][x][0] >100 and img[y][x][1] < 50 and img[y][x][2] < 50)

    # Get Z and Area approximations
    area_temp = np.sum(red_map)

    if area_temp ==0:
        red_m = [res[0]/2-1, res[0]/2-1]
    else:
        red_mul = np.dot(np.vstack((red_map, red_map.T)), np.arange(0, res[0]))/area_temp
        red_m = np.array([np.sum(red_mul[res[0]:2*res[0]]), np.sum(red_mul[0:res[0]])])

    e_ = np.array([red_m[1]-(res[1]-1)/2,       # in uav frame, pixels
                   red_m[0]-(res[0]-1)/2])

    
    if step == 1:
        area0 = area_temp
        area = area0
        Z = 0.719
    else:
        phi = np.arctan(error / e[2])
        area_temp = area_temp / cos(phi)
        area = area + (area_temp-area)*(area<area_temp)
        Z_temp = 0.719*(area0/area)**(1/2)-0.05
        Z = Z + (Z_temp-Z) * (Z>Z_temp)


    # Z = xyz[2] - 0.281

    if step == 1:
        f_x = Z * abs(e_[0] / p_vhc_0[0])
        f_y = Z * abs(e_[1] / p_vhc_0[1])

    x = -e_[0] * Z / f_x
    y = -e_[1] * Z / f_y
    z = (-Z - R[2][0]*x-R[2][1]*y)/R[2][2]

    e_uav = np.array([x, y, z])                 # in uav frame, meters
    e = np.dot(R, e_uav)                        # in wld frame, meters

    error = np.linalg.norm(e[0:2])
    
    # Ascending velocity damping factor
    ad_z = 1 - 1/(1+np.exp(-0.00001 * error))

    # IBVS action
    # vx = ((lamb*Z) - vz)*(e[0]/f_x)
    # vy = ((lamb*Z) - vz)*(e[1]/f_y)

    # vz = vz * ad_z
    vx = lamb*e[0] - vz * e[0]/e[2]
    vy = lamb*e[1] - vz * e[1]/e[2]


    action = np.array([vx, vy, vz])

    zs.append(Z)
    zsm.append(xyz[2]-0.281)
    ars.append(area)
    ers.append(error)
    e_x.append(abs(e[0]))
    e_y.append(abs(e[1]))
    e_z.append(abs(e[2]))

    pybullet.addUserDebugLine(xyz, xyz + e, lifeTime=0.1)
    pybullet.addUserDebugLine(xyz, xyz + e_uav, lifeTime=0.1, lineColorRGB = [0,0,0])
    pybullet.addUserDebugLine(xyz, xyz + 2 * action, lifeTime=0.1, lineColorRGB = [128,0,128])

    # Mark the found points
    img[round(red_m[0])][round(red_m[1])] = [255, 255, 255, 255]
    img[round(res[0]/2-1)][round(res[1]/2-1)] = [128, 0, 128, 255]

    im = Image.fromarray(img, 'RGBA')
    im.save("test_images/drone_view_{0}.png".format(step))
    if step % result_freq == 0:
        print("Step:", step, "| Area:", area, " | Z:", "%.3f" % Z, "| Action:", "%.7f" %action[0],"%.7f" %action[1],"%.7f" %action[2], "\n\n")


    if xyz[2] < 0.01:
        exit()


plt.subplot(2, 2, 1)
plt.plot(zs, label='Z')
plt.plot(zsm, label='Real Z')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(ars, label='Area')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(ers, label='Error')
plt.legend()

plt.subplot(2,2,4)
plt.plot(e_x, label='e_x')
plt.plot(e_y, label='e_y')
plt.plot(e_z, label='e_z')
plt.legend()

plt.show()