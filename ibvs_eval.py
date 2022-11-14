import gym
import os
import numpy as np
from math import cos, sin, sqrt
import matplotlib.pyplot as plt
from gym_pybullet_drones.envs.single_agent_rl.TakeoffAviary import TakeoffAviary
from scipy.spatial.transform import Rotation
from PIL import Image
import copy
import pybullet
seed = 123456
env = gym.make('landing-aviary-v0')#, gui = True)
#pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_SHADOWS,0)

os.system('rm -rf /home/user/landing/landing_rl/test_images')
os.system('mkdir /home/user/landing/landing_rl/test_images')
uav_vels = []
uav_poss = []
Landing_flags = []
Landing_x_y_errors = []
episodes = 100
for jj  in range(episodes):
    print(jj)
    env.seed(seed+jj)
    episode_reward = 0
    done = False
    eval_steps = 0

    max_step = 900

    state=env.reset()
    action = [0, 0, 0]

    zs = []
    zsm=[]
    e_x = []
    e_y = []

    lamb = 0.1
    # Initial position of the vehicle
    p_vhc_0 = np.array([env.pos[0,0], env.pos[0,1], 0.281])
    drone_height_init = env.pos[0,2]
    Z = drone_height_init-p_vhc_0[2]

    for step in range(max_step):
        # Take the step and extract data
        img, reward, done, info = env.step(action)
        res = np.array(img.shape[0:2])
        xyz = info["drone_state"][0]
        if done == True:
            print('episode done')
            break
        # Rotation Matrix from the world to drone frame
        R = Rotation.from_quat(info["drone_state"][2]).as_matrix()
        
        # Get the map of the red area
        red_map = np.zeros((res[0],res[1]))
        for x in range(0, res[1]):
            for y in range(0, res[0]):
                red_map[y][x] = ((img[y][x][0] > 100 and img[y][x][1] < 50 and img[y][x][2] < 50) or 
                                (img[y][x][0] < 50  and img[y][x][1] < 50 and img[y][x][2] < 50))

        # Estimate pad coordinate and Z
        area_temp = np.sum(red_map)

        red_mul = np.dot(np.vstack((red_map, red_map.T)), np.arange(0, res[0]))/area_temp
        red_m = np.array([np.sum(red_mul[res[0]:2*res[0]]), np.sum(red_mul[0:res[0]])])

        e_ = (res-1)/2-red_m    # pixel frame

        if step == 0:
            area0 = area_temp
            area = area0
            # Get focal lengths
            f_x = abs(Z * e_[0] / p_vhc_0[0])
            f_y = abs(Z * e_[1] / p_vhc_0[1])

        if area_temp > area:
            area = area_temp
            #Z = drone_height_init*(area0/area)**(1/2)-0.05
            Z = env.pos[0,2] -0.281


        # Get error in uav and world frames, in meters
        if f_x == 0:
            x = 0
            y = e_[1] * Z / f_y
        elif f_y == 0:
            y = 0 
            x = e_[0] * Z / f_x
        else:
            x = e_[0] * Z / f_x
            y = e_[1] * Z / f_y
        z = (-Z - R[2][0]*x-R[2][1]*y)/R[2][2]
        
        e_uav = np.array([x, y, z])
        e = np.dot(R, e_uav)
        e = e_uav
        # Ascending velocity damping factor
        ad_z = 1 - 1/(1+np.exp(-0.01 * np.linalg.norm(e[0:2])))
        if Z < 5:
            #print(Z)
            lamb = 0.15
        if Z < 3:
            #print(Z)
            lamb = 0.2
        if Z < 1:
            #print(Z)
            lamb = 0.4
        vz = -0.55# * e[2] - 0.15 * ad_z
        vx = lamb*e[0] - vz * e[0]/Z# + 0.2
        vy = lamb*e[1] - vz * e[1]/Z
        #vz = 0
        #vy = 0

        action = np.array([vx, vy, vz])

        zs.append(Z)
        zsm.append(xyz[2]-0.281)
        e_x.append(e[0])
        e_y.append(e[1])

        pybullet.addUserDebugLine(xyz, xyz + action, lifeTime=0.1,lineColorRGB = [150,100,0])
        pybullet.addUserDebugLine(xyz, xyz + e, lifeTime=0.1,lineColorRGB = [0,0,0])
        pybullet.addUserDebugText('{}'.format(step), (1,1,0), textColorRGB = (0,0,0),lifeTime  =0.1)
        if area_temp <= area0*0.8:
            action = [0,0,1]
        if Z < 0.5:
            action[2] = -1
        # Mark the found points
        #img[round(red_m[0])][round(red_m[1])] = [255, 255, 255, 255]
        #img[round(res[0]/2-1)][round(res[1]/2-1)] = [128, 0, 128, 255]
    Landing_flags.append(info["landing"])
    x_y_landing_error = [info["x error"],info["y error"]]
    print(x_y_landing_error)
    Landing_x_y_errors.append(copy.deepcopy(x_y_landing_error))
#plot success ratio
Landing_flags = np.array(Landing_flags)
successes = np.sum(Landing_flags)/episodes
print(successes)
plt.figure(1)
approaches = ['SAC (ours)']
success_rates = [successes]
plt.bar(approaches,success_rates)
plt.xlabel('approaches')
# naming the y axis
plt.ylabel('Success Rate')
#compute x-y landing errors
Landing_x_y_errors = np.array(Landing_x_y_errors)
landing_error = np.mean(Landing_x_y_errors, axis = 0)
#print(Landing_x_y_errors)
print(landing_error)
plt.show()
env.close()

