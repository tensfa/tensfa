def resolve_shape(v):
    if hasattr(v, 'shape'):
        s = v.shape
        import tensorflow as tf
        if isinstance(s, tf.TensorShape):
            if s.dims:
                return s.as_list()
            else:
                return [1]
        else:
            return list(s)
    elif hasattr(v, '__len__'):
        return [resolve_shape(i) for i in v]
    elif isinstance(v, (int, float)):
        return [1]
    else:
        return ['unknown']

def find_parameter_end(line_sources, start_lineno, start_index):
    stack = []
    single_quote = False
    double_quote = False
    for i, line in enumerate(line_sources[start_lineno-1:], start=start_lineno-1):
        if i != start_lineno-1:
            start_index = 0
        for j, c in enumerate(line_sources[i][start_index:], start=start_index):
            if c == '\'':
                if not double_quote:
                    single_quote = not single_quote
            elif c == '\"':
                if not single_quote:
                    double_quote = not double_quote
            elif not single_quote and not double_quote:
                if c in ['(', '[', '{']:
                    stack.append(c)
                elif c in [']', '}']:
                    stack.pop()
                elif c == ')':
                    if len(stack) == 0:
                        return i+1, j
                    else:
                        stack.pop()
                elif c == ',':
                    if len(stack) == 0:
                        return i+1, j

def split_dict_string(dict_s):
    parsed_dict = {}
    key, value = None, None
    start, end = None, None

    stack = []
    single_quote = False
    double_quote = False
    for i, c in enumerate(dict_s):
        if c == '\'':
            if not double_quote:
                single_quote = not single_quote
        elif c == '\"':
            if not single_quote:
                double_quote = not double_quote
        elif not single_quote and not double_quote:
            if c in ['(', '[']:
                stack.append(c)
            elif c in [')', ']']:
                stack.pop()
            elif c == '{':
                if len(stack) == 0:
                    start = i + 1
                else:
                    stack.append(c)
            elif c == ':':
                if len(stack) == 0:
                    end = i
                    key = dict_s[start:end].strip()
                    start = i + 1
            elif c == ',':
                if len(stack) == 0:
                    end = i
                    value = dict_s[start:end].strip()
                    parsed_dict[key] = value
                    start = i + 1
            elif c == '}':
                if len(stack) == 0:
                    end = i
                    value = dict_s[start:end].strip()
                    parsed_dict[key] = value
                else:
                    stack.pop()
    return parsed_dict

def split_tuple_or_list_string(s):
    s = s.strip()[1:-1]
    list_tuple = []
    start, end = 0, None

    stack = []
    single_quote = False
    double_quote = False
    for i, c in enumerate(s):
        if c == '\'':
            if not double_quote:
                single_quote = not single_quote
        elif c == '\"':
            if not single_quote:
                double_quote = not double_quote
        elif not single_quote and not double_quote:
            if c in ['(', '[', '{']:
                stack.append(c)
            elif c in [')', ']', '}']:
                stack.pop()
            elif c == ',':
                if len(stack) == 0:
                    end = i
                    value = s[start:end].strip()
                    if value:
                        list_tuple.append(value)
                    start = i + 1

    value = s[start:].strip()
    if value:
        list_tuple.append(value)
    return list_tuple
