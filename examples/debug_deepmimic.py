from pybullet_utils import bullet_client
import pybullet as p
import pybullet_data

import math
BASE_LINK_ID = -1

_client = bullet_client.BulletClient(connection_mode=p.GUI)
_client.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
_client.setAdditionalSearchPath("./data")
#_client.setAdditionalSearchPath(pybullet_data.getDataPath())

z2y = p.getQuaternionFromEuler([-math.pi * 0.5, 0, 0])

robotId = _client.loadURDF(
    "humanoid.urdf",
    [0, 0, 0], 
    globalScaling=0.25
)
num_joints = p.getNumJoints(robotId)
jointIndices = range(num_joints)


def _record_masses(model):
    num_joints = p.getNumJoints(model)
    jointIndices = range(num_joints)

    infos = p.getDynamicsInfo(model, jointIndices)
    masses = [x[0] for x in infos]
    return masses

def calcCOM(model):
    num_joints = p.getNumJoints(model)
    jointIndices = range(num_joints)
    link_states = p.getLinkStates(model, jointIndices)
    for i, s in enumerate(link_states):
    [s[0] for o in link_states]

masses = _record_masses(robotId)
total_mass = sum(masses)

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

