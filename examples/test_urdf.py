from pybullet_utils import bullet_client
import pybullet as p
import pybullet_data

import math

_client = bullet_client.BulletClient(connection_mode=p.GUI)
_client.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
_client.setAdditionalSearchPath("./data")
#_client.setAdditionalSearchPath(pybullet_data.getDataPath())

z2y = p.getQuaternionFromEuler([-math.pi * 0.5, 0, 0])

robotId = _client.loadURDF(
    "big_test.urdf",
    [0, 0, 0],
    #globalScaling=0.25
)
num_joints = p.getNumJoints(robotId)
jointIndices = range(num_joints)

while True:
    _client.stepSimulation()
    simJointStates = p.getJointStatesMultiDof(
        robotId, jointIndices)

    print("joint states", simJointStates)
    link_states = p.getLinkStates(robotId, jointIndices)
    print("link states")
    for i, a in enumerate(link_states):
        print("com: ", a[0])
        info = p.getJointInfo(robotId, i)
        print(info)
    #break
