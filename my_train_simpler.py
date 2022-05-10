import gym
import os
import numpy as np
import matplotlib.pyplot as plt
from gym_pybullet_drones.envs.single_agent_rl.TakeoffAviary import TakeoffAviary
from PIL import Image

env = 'landing-aviary-v0'
env = gym.make(env)
os.system('rm -rf /home/user/landing/landing_rl/test_images')
os.system('mkdir /home/user/landing/landing_rl/test_images')

res = np.array([64,64])
max_step = 250
step = 0
zs = []
zsm=[]
ars = []
ers = []

state=env.reset()
action = [0.1, 0, 0]

v_veh = np.array([0, 0])
v_vehicle = 0.25*np.tile(v_veh, 5)
v_z = 0.3

lamb = 0.001
f =  3 ** 0.5


for step in range(max_step):
    step+=1
    
    next_state, reward, done, info = env.step(action)
    img = next_state.transpose(1, 2, 0)
    
    # Get the map of red area
    red_map = np.zeros(res)
    for x in range(0, res[1]):
        for y in range(0, res[0]):
            red_map[y][x] = (img[y][x][0] >100 and img[y][x][1] < 50 and img[y][x][2] < 50)

    if step == 1:
        area0 = np.sum(red_map)
        area = area0
        Z = 0.719

    area_temp = np.sum(red_map)
    if area_temp ==0:
        area_temp = 1
    area = area + (area_temp-area)*(area<area_temp)

    Z_temp = 0.719*(area0/area)**(1/2)-0.05
    Z = Z + (Z_temp-Z)*(Z>Z_temp)

    red_mul = np.matmul(np.vstack((red_map, red_map.T)), np.arange(0, res[1]))/area_temp
    red_m = np.int_([np.sum(red_mul[res[1]:2*res[1]]), np.sum(red_mul[0:res[1]])])

    # Set the desired positions
    des_m = np.int_(res/2-[1,1])

    # Marking the found points
    img[red_m[0]][red_m[1]] = [255, 255, 255, 255]
    img[des_m[0]][des_m[1]] = [128, 0, 128, 255]
    
    x = red_m[1]
    y = red_m[0]
    e = des_m - red_m
    # e = np.array([e[1], e[0]])

    vz = 0
    vx = 100*((lamb*Z) - vz)*(e[1]/f) 
    vy = 100*((lamb*Z) - vz)*(e[0]/f) 

    action = [lamb*vx, -lamb*vy, vz]

    error = np.linalg.norm(e)
    # ad_z = 1 - 1/(1+np.exp(-0.01 * error))
    # action[2] = ad_z * action[2]

    zs.append(Z)
    ars.append(area)
    ers.append(error)

    im = Image.fromarray(img, 'RGBA')
    im.save("test_images/drone_view_{0}.png".format(step))
    print("Step:", step, "| Area:", area, "| Z:", "%.3f" % Z, "| Action:", "%.7f" %action[0],"%.7f" %action[1],"%.7f" %action[2], "\n")

plt.subplot(3, 1, 1)
plt.plot(zs, label='Z')
plt.legend()
plt.subplot(3, 1, 2)
plt.plot(ars, label='Area')
plt.legend()
plt.subplot(3, 1, 3)
plt.plot(ers, label='Error')
plt.legend()
plt.show()