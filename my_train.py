import gym
import numpy as np
import matplotlib.pyplot as plt
from gym_pybullet_drones.envs.single_agent_rl.TakeoffAviary import TakeoffAviary
from PIL import Image

env = 'landing-aviary-v0'
env = gym.make(env)

res = np.array([64,64])
max_step = 250
step = 0
zs = []
ars = []
ers = []

state=env.reset()
action = [0, 0, 0]

for step in range(max_step):
    step+=1
    
    next_state, reward, done, info = env.step(action)
    img = next_state.transpose(1, 2, 0)
    print(info)
    
    # Get the map of red area
    red_map = np.zeros(res)
    for x in range(0, res[1]):
        for y in range(0, res[0]):
            red_map[y][x] = (img[y][x][0] >100 and img[y][x][1] < 50 and img[y][x][2] < 50)

    area = np.sum(red_map)

    if step == 1:
        area0 = area
    
    if area < area0:
        area = area0
        red_map[32][32] = 1

    red_mul = np.matmul(np.vstack((red_map, red_map.T)), np.arange(0, res[1]))/area
    red_m = np.int_([np.sum(red_mul[res[1]:2*res[1]]), np.sum(red_mul[0:res[1]])])

    # Find the left and rightmost red points
    found = False
    red_l = red_m
    red_r = red_m
    for x in range(0, res[1]-1):
        y = red_m[0]
        if (img[y][x][0] > 150 and img[y][x][1] < 100):
            red_r = np.array([y, x])
            if not found:
                red_l = np.array([y, x])
                found = True

    # Find the upper and bottommost red points
    found = False
    red_u = red_m
    red_b = red_m
    for y in range(0, res[0]-1):
        x = red_m[1]
        if (img[y][x][0] > 150 and img[y][x][1] < 100):
            red_b = np.array([y, x])
            if not found:
                red_u = np.array([y, x])
                found = True

    # Get the radius in each direction
    r_r = red_r[1] - red_m[1]
    r_l = red_m[1] - red_l[1]
    r_u = red_m[0] - red_u[0]
    r_b = red_b[0] - red_m[0]

    # Set the desired positions
    des_m = np.int_(res/2-[1,0])
    des_r = des_m + [0, r_r]
    des_l = des_m - [0, r_l]
    des_u = des_m - [r_u, 0]
    des_b = des_m + [r_b, 0]

    # Marking the found points
    img[red_m[0]][red_m[1]] = [255, 255, 255, 255]
    img[red_r[0]][red_r[1]] = [255, 255, 255, 255]
    img[red_l[0]][red_l[1]] = [255, 255, 255, 255]
    img[red_u[0]][red_u[1]] = [255, 255, 255, 255]
    img[red_b[0]][red_b[1]] = [255, 255, 255, 255]

    img[des_m[0]][des_m[1]] = [128, 0, 128, 255]
    img[des_r[0]][des_r[1]] = [128, 0, 128, 255]
    img[des_l[0]][des_l[1]] = [128, 0, 128, 255]
    img[des_u[0]][des_u[1]] = [128, 0, 128, 255]
    img[des_b[0]][des_b[1]] = [128, 0, 128, 255]

    Z = 0.8*(area0/area)**(1/2) - 0.143
    Z = info[2]
    # Z = 0.7187*(area0/area)**(1/2)
    # 0.2813

    if Z < 0.003:
        Z = 0.003

    lamb = 0.01
    f =  3 ** 0.5

    # Defining the errors
    e = np.vstack((des_m - red_m,
                   des_r - red_r,
                   des_l - red_l,
                   des_u - red_u,
                   des_b - red_b))

    e = np.matmul(e, [[0, 1], [1, 0]]).flatten()

    Ls = np.array([[-f, 0, red_m[1]], 
                   [0, -f, red_m[0]],

                   [-f, 0, red_r[1]], 
                   [0, -f, red_r[0]], 
                   
                   [-f, 0, red_l[1]], 
                   [0, -f, red_l[0]],

                   [-f, 0, red_u[1]], 
                   [0, -f, red_u[0]],

                   [-f, 0, red_b[1]], 
                   [0, -f, red_b[0]]]) / Z

    L_ep = np.linalg.pinv(Ls)

    action = np.dot(L_ep, (-lamb * e.T))

    error = np.linalg.norm(e)

    ad_z = 1 - 1/(1+np.exp(-0.001 * error))
    action[0] += 0.65*0
    action[2] = ad_z * (action[2] - 0.2)

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