import json
import os
import sys
from contextlib import contextmanager
from collections import namedtuple, defaultdict

import tensorflow as tf

VarKey = namedtuple('VarKey', ['path', 'lineno', 'var_name'])
running = False

TOP_DIR = os.path.join(os.getcwd(), '')
TOP_DIR_DOT = os.path.join(TOP_DIR, '.')
TOP_DIR_LEN = len(TOP_DIR)

@contextmanager
def collect():
    start()
    try:
        yield
    finally:
        stop()

def pause():
    return stop()

def stop():
    global running
    running = False
    # stop_shapes_collection()

def resume():
    return start()

def start():
    global running
    running = True

def default_filter_filename(filename):
    if filename is None:
        return None
    elif filename.startswith(TOP_DIR):
        if filename.startswith(TOP_DIR_DOT):
            return None
        else:
            return filename[TOP_DIR_LEN:].lstrip(os.sep)
    elif filename.startswith(os.sep):
        return None
    else:
        return filename

_filenames = []
_filter_filename = default_filter_filename

_var_dict = {}
var_stack = [{}]

last_lineno = -1
_lineno_varname = {}
_trace_shape = False

f_globals, f_locals = {}, {} # context
def _trace_dispatch(frame, event, arg):
    global last_lineno, f_globals, f_locals
    if not running:
        return
    code = frame.f_code
    filename = _filter_filename(code.co_filename)
    if filename not in _filenames:
        return

    now_lineno = frame.f_lineno
    frame.f_trace = _trace_dispatch

    if event == 'line':
        if frame.f_globals != {}:
            f_globals = frame.f_globals
        if frame.f_locals != {}:
            f_locals = frame.f_locals
    elif event == 'call':
        # print(f'call at {now_lineno}')
        var_stack.append({})
    elif event == 'return':
        # print(f'return at {now_lineno}')
        var_stack.pop()
    else:
        return

    if filename and _trace_shape:
        def get_shape(v):
            if hasattr(v, 'shape'):
                s = v.shape
                if isinstance(s, tf.TensorShape):
                    if s.dims:
                        return s.as_list()
                    else:
                        return [1]
                else:
                    return list(s)
            # elif hasattr(v, '__len__'):
            #     return [get_shape(i) for i in v]
            elif isinstance(v, (int, float)):
                return [1]
            else:
                return ['unknown']

        if event == 'line':
            var_stack[-1] = frame.f_locals
            if last_lineno in _lineno_varname:
                for var_name in _lineno_varname[last_lineno]:
                    for i in range(len(var_stack)-1, -1, -1):
                        if var_name in var_stack[i]:
                            var = var_stack[i][var_name]
                            var_key = VarKey(filename, last_lineno, var_name)
                            try:
                                _var_dict[var_key] = get_shape(var)
                            except Exception as e:
                                print(e)
                            break
            last_lineno = frame.f_lineno
        elif event == 'call':
            pass
        elif event == 'return':
            if last_lineno in _lineno_varname:
                for var_name in _lineno_varname[last_lineno]:
                    for i in range(len(var_stack)-1, -1, -1):
                        if var_name in var_stack[i]:
                            var = var_stack[i][var_name]
                            var_key = VarKey(filename, last_lineno, var_name)
                            try:
                                _var_dict[var_key] = get_shape(var)
                            except Exception as e:
                                print(e)
                            break

def get_context():
    return f_globals, f_locals

def _dump_impl():
    res = defaultdict(list)
    for var_key, shape in _var_dict.items():
        res[var_key.lineno].append({
                'path': var_key.path,
                'lineno': var_key.lineno,
                'var_name': var_key.var_name,
                'shape': shape
            }
        )
    return res

def dump_stats(filename):
    res = _dump_impl()
    f = open('%s_shape.json' % filename, 'w')
    json.dump(res, f, indent=4)
    f.close()

def dumps_stats():
    res = _dump_impl()
    return res

def init_shapes_collection(filenames, lineno_varname, filter_filename=default_filter_filename, trace_shape=False):
    global _filenames, _filter_filename, _lineno_varname, _trace_shape
    _filenames = filenames
    _filter_filename = filter_filename
    _lineno_varname = lineno_varname
    _trace_shape = trace_shape

    sys.settrace(_trace_dispatch)
    # sys.setprofile(_trace_dispatch)

def stop_shapes_collection():
    sys.settrace(None)
    # sys.setprofile(None)
