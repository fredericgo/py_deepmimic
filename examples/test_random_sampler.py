from py_deepmimic.env.deepmimic_gym_env import DeepMimicGymEnv
import numpy as np
import pkg_resources

backflip_file = pkg_resources.resource_filename(
    "py_deepmimic",
    "data/args/train_humanoid3d_backflip_args.txt"
)
env = DeepMimicGymEnv(backflip_file, enable_draw=True)
#env.set_mode(Mode.TEST)
state = env.reset()
for i in range(100):
    c = 1
    while True:
        c += 1
        a = env.action_space.sample()
        state, rew, done, _ = env.step(a)
        print(rew)
        #env.render()
        if done:
            print(c)
            env.set_sample_count(i*100000)
            state = env.reset()
            break
