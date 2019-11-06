import numpy as np
import copy
import random
# update t = sample_count / num_anneal
# pow 4

# timer params lim: min, max, exp
# timer params end: min, max, exp
# anneal samples 
# -> t = clamp(sample_count / anneal_samples, 0, 1)**4
# lerp blend: 
# curr_params blend lerp 


# params
class TimerParams:
    def __init__(self, min, max):
        self.min = min
        self.max = max
    
    def blend(self, other, t):
        self.min = np.interp(t, [0, 1], [self.min, other.min])
        self.max = np.interp(t, [0, 1], [self.max, other.max])


class Timer:
    def __init__(self, 
            annealing_samples,
            time_lim_min,
            time_lim_max,
            time_end_min,
            time_end_max):
        self.annealing_samples = annealing_samples
        self.begin_params = TimerParams(time_lim_min, time_lim_max)
        self.curr_params = None
        self.end_params = TimerParams(time_end_min, time_end_max)
        self.reset_params()
        self.max_time = None
        self.t = 0

    def update(self, timestep):
        self.t += timestep

    def reset(self):
        self.t = 0
        self.max_time = random.uniform(self.curr_params.min, self.curr_params.max)

    def reset_params(self):
        self.curr_params = copy.deepcopy(self.begin_params)

    def set_sample_count(self, sample_count):
        t = sample_count / float(self.annealing_samples)
        t = np.clip(t, 0., 1.)
        lerp = t**4
        self.curr_params.blend(self.end_params, lerp)
    
    def is_end(self):
        return self.t > self.max_time


if __name__ == "__main__":
    timer = Timer(32000000, 0.5, 0.5, 20, 20)
    for i in range(0, 32000000, 100000):
        timer.set_sample_count(i)

