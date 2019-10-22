import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data
import time

direct = p.connect(p.GUI)  # , options="--window_backend=2 --render_device=0")
#egl = p.loadPlugin("eglRendererPlugin")
p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.loadURDF('plane.urdf')
p.loadURDF("r2d2.urdf", [0, 0, 1])

width = 240
height = 240

fov = 60
aspect = width / height
near = 0.02
far = 1

view_matrix = p.computeViewMatrix([0, 0, 0.5], [0, 0, 0], [1, 0, 0])
projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

# Get depth values using the OpenGL renderer
images = p.getCameraImage(width,
                          height,
                          view_matrix,
                          projection_matrix,
                          shadow=True,
                          renderer=p.ER_BULLET_HARDWARE_OPENGL)
rgb_opengl = np.reshape(images[2], (height, width, 4)) * 1. / 255.
depth_buffer_opengl = np.reshape(images[3], [width, height])
depth_opengl = far * near / (far - (far - near) * depth_buffer_opengl)
seg_opengl = np.reshape(images[4], [width, height]) * 1. / 255.
time.sleep(1)

viewMat = [
    0.642787516117096, -0.4393851161003113, 0.6275069713592529, 0.0, 0.766044557094574,
    0.36868777871131897, -0.5265407562255859, 0.0, -
    0.0, 0.8191521167755127, 0.5735764503479004,
    0.0, 2.384185791015625e-07, 2.384185791015625e-07, -5.000000476837158, 1.0
]
projMat = [
    0.7499999403953552, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -
    1.0000200271606445, -1.0,
    0.0, 0.0, -0.02000020071864128, 0.0
]
images = p.getCameraImage(width,
                          height,
                          viewMatrix=viewMat,
                          projectionMatrix=projMat,
                          renderer=p.ER_BULLET_HARDWARE_OPENGL,
                          flags=p.ER_USE_PROJECTIVE_TEXTURE,
                          projectiveTextureView=viewMat,
                          projectiveTextureProj=projMat)
proj_opengl = np.reshape(images[2], (height, width, 4)) * 1. / 255.

while True:
    p.stepSimulation()


