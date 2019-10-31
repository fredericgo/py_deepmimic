from py_deepmimic.env import argparser
arg_file = "train_humanoid3d_walk_args.txt"

config = argparser.load_file(arg_file)
print(config)