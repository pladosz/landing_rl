
import pybullet as p
import time
import pybullet_data
import os
import numpy as np


# ~~~~~~~~~~~~~~~~~~~~~

Velocity = 10200
Joint = [1, 4]        #1, 4
force_limit = 600

startTime = 10
stopTime = 26000

xacro_file = "/home/user/landing/landing_rl/g_vehicle/car_v3.urdf"
urdf_file = "/home/user/landing/landing_rl/g_vehicle/parsed2.urdf"

# ~~~~~~~~~~~~~~~~~~~~~



parser_command = 'xacro ' + xacro_file + ' > ' + urdf_file

os.system(parser_command)       # Parse the xacro file to a new urdf file


physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf")
boxId = p.loadURDF(urdf_file)


# print(p.getNumJoints(boxId))
# print(p.getJointInfo(boxId, Joint[1]))
print(p.getJointInfo(boxId, 8))
# print(p.getLinkState(boxId, 9)[0:2])


for i in range (startTime):
        p.stepSimulation()
        time.sleep(1./240.)


# p.setJointMotorControl2(bodyUniqueId=boxId, jointIndex=7, controlMode=p.VELOCITY_CONTROL, targetVelocity=200, force = 99999)
p.setJointMotorControlMultiDof(boxId, 8, p.POSITION_CONTROL, targetPosition = [0.1*np.sin(i),0,0.1,1])

# p.setJointMotorControl2(bodyUniqueId=boxId, jointIndex=Joint[1], controlMode=p.VELOCITY_CONTROL, targetVelocity=Velocity, force=force_limit)


for i in range (stopTime-startTime):
    p.stepSimulation()
    time.sleep(1./240.)
    p.setJointMotorControlMultiDof(boxId, 8, p.POSITION_CONTROL, targetPosition = [0.1*np.sin(np.deg2rad(i)),0.1*np.cos(np.deg2rad(i)-0.35),0.1*np.sin(0.1+np.deg2rad(i)),1])
    # print(p.getBasePositionAndOrientation(boxId))

cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
# print(cubePos,cubeOrn)
p.disconnect()



        # roll = 0.01 * np.sin(np.deg2rad(self.step_counter))
        # pitch = 0.01 * np.cos(np.deg2rad(self.step_counter ** 0.5)-0.1)
        # yaw = 0.1 * (roll * pitch) ** 0.5
        # p.setJointMotorControlMultiDof(self.vehicleId, 8, p.POSITION_CONTROL, targetPosition = [roll, pitch, yaw, 1])