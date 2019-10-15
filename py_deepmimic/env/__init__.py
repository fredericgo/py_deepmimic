from py_deepmimic.env import deepmimic_gym_env


def make(env_id, **kwargs):
    domain, task = env_id.split("-")
    env_string = "{}{}Env".format(domain.capitalize(), task.capitalize())
    env_cls = getattr(deepmimic_gym_env, env_string)
    print(kwargs)
    return env_cls(**kwargs)
