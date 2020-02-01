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

_planeId = p.loadURDF("plane_implicit.urdf", [0, 0, 0],
                                               z2y,
                                               useMaximalCoordinates=True)
robotId = p.loadURDF(
    "humanoid.urdf",
    [0, 0.85, 0], 
    globalScaling=0.25
)

_ballUniqueId = p.loadURDF("sphere2.urdf", [2, .5, 2])

num_joints = p.getNumJoints(robotId)
jointIndices = range(num_joints)

p.setGravity(0, -9.8, 0)

def getBallCOM(ballId):
    p.getLinkState(ballId)
while True:
    p.stepSimulation()
    s = p.getBasePositionAndOrientation(_ballUniqueId)
    h = p.getBasePositionAndOrientation(robotId)

    print(s)
    print(h)

    

