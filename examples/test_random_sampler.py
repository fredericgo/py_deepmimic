from py_deepmimic.env.deepmimic_gym_env import DeepMimicGymEnv
import numpy as np

arg_file = "args/train_humanoid3d_backflip_args.txt"

env = DeepMimicGymEnv(arg_file)
#env.set_mode(Mode.TEST)
state = env.reset()
for i in range(100):
    while True:
        a = env.action_space.sample()
        state, rew, done, _ = env.step(a)
        #env.render()
        if done:
            state = env.reset()
            break
