import re as RE

KEY_TYPE = {
    'scene': str,
    'time_lim_min': float,
    'time_lim_max': float,
    'time_lim_exp': float,
    'time_end_lim_min': float,
    'time_end_lim_max': float,
    'time_end_lim_exp': float,
    'anneal_samples': int,
    'num_update_substeps': int,
    'num_sim_substeps': int,
    'world_scale': float,
    'terrain_file': str,
    'char_types': str,
    'character_files': str,
    'enable_char_soft_contact': bool,
    'enable_root_rot_fail': bool,
    'fall_contact_bodies': int,
    'char_ctrls': str,
    'char_ctrl_files': str,
    'motion_file': str,
    'sync_char_root_pos': bool,
    'sync_char_root_rot': bool,
    'agent_files': str,
    'output_path': str
}

def _is_comment(str):
    is_comment = False
    if (len(str) > 0):
      is_comment = str[0] == '#'
    return is_comment

def _is_key(str):
    is_key = False
    if (len(str) >= 3):
        is_key = str[0] == '-' and str[1] == '-'
    return is_key

def load_args(arg_strs):
    _table = dict()
    vals = []
    curr_key = ''

    for str in arg_strs:
      if not (_is_comment(str)):
        is_key = _is_key(str)
        if (is_key):
            if (curr_key != ''):
                if (curr_key not in _table):
                    _table[curr_key] = [KEY_TYPE[curr_key](v) for v in vals]

            vals = []
            curr_key = str[2::]
        else:
            vals.append(str)

    if (curr_key != ''):
        if (curr_key not in _table):
            _table[curr_key] = vals

        vals = []
    return _table

def load_file(filename):
    with open(filename, 'r') as file:
        lines = RE.split(r'[\n\r]+', file.read())
        file.close()
        arg_strs = []
        for line in lines:
            if (len(line) > 0 and not _is_comment(line)):
                arg_strs += line.split()

        return load_args(arg_strs)
