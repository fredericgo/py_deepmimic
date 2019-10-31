from pybullet_utils import bullet_client
import pybullet as p
import pybullet_data

import math

_client = bullet_client.BulletClient(connection_mode=p.GUI)
_client.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
_client.setAdditionalSearchPath("./data")
#_client.setAdditionalSearchPath(pybullet_data.getDataPath())

z2y = p.getQuaternionFromEuler([-math.pi * 0.5, 0, 0])

_planeId = _client.loadURDF(
    "ground.urdf",
    [15, 0, 0]
)

while True:
    _client.stepSimulation()
