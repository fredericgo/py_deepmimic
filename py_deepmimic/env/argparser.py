import re as RE

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
                    _table[curr_key] = vals

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
