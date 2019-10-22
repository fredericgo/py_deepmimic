from .deepmimic_gym_env import DeepMimicGymEnv

import numpy as np
import pkg_resources


class HumanoidRadiusEnv(DeepMimicGymEnv):
    _file = pkg_resources.resource_filename(
        "py_deepmimic",
        "data/args/train_humanoid3d_walk_args.txt"
    )

    def __init__(self,
                 enable_draw=False,
                 **kwargs):
        super(HumanoidRadiusEnv, self).__init__(
            self._file, enable_draw=enable_draw, **kwargs)
    
    def reset(self):
        startTime = 0

        self.t = startTime
        self._humanoid.setSimTime(startTime)
        self._humanoid.resetPoseWithoutVelocity()
        #this clears the contact points. Todo: add API to explicitly clear all contact points?
        #self._pybullet_client.stepSimulation()
        self._humanoid.resetPoseWithoutVelocity()
        self.needs_update_time = self.t - 1  # force update
        return self.observations()

    def calc_reward(self):
        radius = 5
        x = self._humanoid.getPosition()
        r = np.sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2])
        reward = r / radius
        if r > 5:
            reward = 0
        return reward
