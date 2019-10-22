from py_deepmimic.env import humanoid_stable_pd
from py_deepmimic.env import motion_capture_data
from py_deepmimic.env import argparser

from pybullet_utils import bullet_client
import pybullet_data
import pybullet as p1
import numpy as np

import pkg_resources
import math
import time
import os

arg_file = _file = pkg_resources.resource_filename(
    "py_deepmimic",
    "data/args/train_humanoid3d_cartwheel_args.txt"
)

def load_mocap_data(filename):
    _mocapData = motion_capture_data.MotionCaptureData()
    motionPath = pybullet_data.getDataPath() + "/" + filename
    _mocapData.Load(motionPath)
    print("motion_file=", filename)
    return _mocapData

def _make_dir(filename):
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)

def save_video(video_frames, filename, fps=60):
    assert fps == int(fps), fps
    import skvideo.io
    _make_dir(filename)
    skvideo.io.vwrite(filename, video_frames,
                      inputdict={'-r': str(int(fps))},
                      outputdict={"-vcodec": "libx264", "-pix_fmt": "yuv420p"})

class Camera:
    def __init__(self):
        self.camTargetPos = [0, 1, 0]

        self.pitch = -20
        self.yaw = 190
        self.roll = 0
        self.upAxisIndex = 1
        self.camDistance = 4

        self.width = 800
        self.height = 600

        self.nearPlane = 0.01
        self.farPlane = 100

        self.fov = 60
    
        self.viewMatrix = p1.computeViewMatrixFromYawPitchRoll(
                            self.camTargetPos, 
                            self.camDistance, 
                            self.yaw, 
                            self.pitch,
                            self.roll, 
                            self.upAxisIndex)
        self.aspect = self.width / self.height
        self.projectionMatrix = p1.computeProjectionMatrixFOV(
                                    self.fov, 
                                    self.aspect, 
                                    self.nearPlane, 
                                    self.farPlane)

class NewEnvironment:
    def __init__(self,
                 arg_file):
        self.timestep = 1 / 240.
        self._config = argparser.load_file(arg_file)
        self.initialized = False

        self._initialize()

    def _initialize(self):
        if not self.initialized:
            self._client = bullet_client.BulletClient(connection_mode=p1.GUI)
            self._client.configureDebugVisualizer(p1.COV_ENABLE_Y_AXIS_UP, 1)
            self._client.configureDebugVisualizer(p1.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
            self._client.configureDebugVisualizer(p1.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
            self._client.configureDebugVisualizer(p1.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)

            self._client.setGravity(0, -9.8, 0)
            self._client.setPhysicsEngineParameter(numSolverIterations=10)
            self._client.setPhysicsEngineParameter(numSubSteps=1)
            self._client.setTimeStep(self.timestep)

            self._client.setAdditionalSearchPath(pybullet_data.getDataPath())
            self._build_plane()

            self._build_humanoid()
            #sphereUid = self._client.loadURDF("sphere2.urdf", [0, 0, 0])
            self._build_camera()

    def _build_plane(self):
        z2y = p1.getQuaternionFromEuler([-math.pi * 0.5, 0, 0])

        self._planeId = self._client.loadURDF(
             "plane_implicit.urdf",
             [0, 0, 0],
             z2y,
             useMaximalCoordinates=True
        )
        self._client.changeDynamics(self._planeId, linkIndex=-1, lateralFriction=0.9)

    def _build_humanoid(self):
        motion_file = self._config["motion_file"][0]
        mocap_data = load_mocap_data(motion_file)

        useFixedBase = False

        self._humanoid = humanoid_stable_pd.HumanoidStablePD(
                            self._client,
                            mocap_data,
                            self.timestep,
                            useFixedBase,
                            self._config
                        )

    def _build_camera(self):
        self._camera = Camera()

    def reset(self):
        start_time = 0
        self._humanoid.setSimTime(start_time)
        self._humanoid.resetPoseWithoutVelocity()
        #self._humanoid.resetPose()

    def capture_image(self):
        image = self._client.getCameraImage(
                    self._camera.width,
                    self._camera.height,
                    self._camera.viewMatrix,
                    self._camera.projectionMatrix,
                    shadow=1,
                    lightDirection=[1, 1, 1],
                    renderer=p1.ER_BULLET_HARDWARE_OPENGL,
                )
        return image
    
    def step(self):
        self._client.stepSimulation()
        r = self.calc_reward()
        print(r)

    def calc_reward(self):
        radius = 5
        x = self._humanoid.getPosition()
        r = np.sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2])
        reward = r / radius
        if r > 5:
            reward = 0
        return reward

env = NewEnvironment(arg_file)

env.reset()
images = []
while True:
    time.sleep(0.01)
    env.step()
    pass
    #image = env.capture_image()
    #images.append(image[2][..., :3])


#images = np.stack(images, axis=0)
#images = images.astype(np.uint8)

#save_video(images, './videos/test.mp4')
