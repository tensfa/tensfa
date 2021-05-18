import math
import re
import random
from utils import find_parameter_end, resolve_shape
random.seed(2021)

re_loss_function = {'mean_squared_error', 'mse',
                    'mean_absolute_error', 'mae',
                    'mean_absolute_percentage_error', 'mape',
                    'mean_squared_logarithmic_error', 'msle'}

def insert_new_lines(source, linenos, lines):
    if not linenos or not lines:
        return source
    linenos, lines = zip(*sorted(zip(linenos, lines), key=lambda x:x[0]))
    new_source = []
    j = 0
    for i, line in enumerate(source):
        while j < len(linenos) and i == linenos[j] - 1:
            new_source.append(lines[j])
            j += 1
        new_source.append(line)
    return new_source

def calculate_blank(source, lineno):
    if 'for' not in source[lineno - 1]:
        blank = source[lineno - 1].replace(source[lineno - 1].lstrip(), '')
    else:
        blank = source[lineno].replace(source[lineno].lstrip(), '')
    return blank

def insert_or_replace(source, var_name, lineno, patch, mode):
    new_linenos, new_lines = [], []
    if mode == 'replace':
        source[lineno - 1] = source[lineno - 1].replace(var_name, patch)
    elif mode == 'insert':
        blank = calculate_blank(source, lineno)
        line = blank + f'{var_name} = {patch}\n'
        new_linenos.append(lineno + 1)
        new_lines.append(line)
    else:
        raise ValueError
    source = insert_new_lines(source, new_linenos, new_lines)
    return source


def find_data_dependency(function_key, raw_param_name, visitor):
    data_dependency = None
    for var in visitor.func_key_var_params[function_key]:
        if var.name == raw_param_name and var in visitor.graph:
            data_dependency = visitor.graph[var][-1]
            break
    return data_dependency


def generate_a_random_patch(var_name, replace):
    replaced_random_patches = [f'np.expand_dims({var_name}, -1)',
                               f'np.expand_dims({var_name}, 0)',
                               f'np.argmax({var_name}, axis=-1)',
                               f'np.argmax({var_name}, axis=0)',
                               f'{var_name}.transpose()']
    line_random_patches = [f'{var_name} = np.expand_dims({var_name}, -1)\n',
                           f'{var_name} = np.expand_dims({var_name}, 0)\n',
                           f'{var_name} = np.argmax({var_name}, axis=-1)\n',
                           f'{var_name} = np.argmax({var_name}, axis=0)\n',
                           f'{var_name} = {var_name}.transpose()\n']
    if replace:
        return random.choice(replaced_random_patches)
    else:
        return random.choice(line_random_patches)

def random_patch(source, except_lineno, data_in_out_name_shape):
    fixed = False
    new_linenos, new_lines = [], []
    for d in data_in_out_name_shape:
        var_name, lineno = d.var.name, d.var.lineno
        if d.replace:
            source[except_lineno - 1] = \
                source[except_lineno - 1].replace(d.var.name, generate_a_random_patch(var_name, d.replace))
        else:
            blank = calculate_blank(source, lineno)
            line = blank + generate_a_random_patch(var_name, d.replace)
            new_linenos.append(lineno + 1)
            new_lines.append(line)
        fixed = True
    source = insert_new_lines(source, new_linenos, new_lines)
    return fixed, source

def expand_x_data_first_dim(source, except_lineno, data_in_out_name_shape, model_in_out_name_shape):
    fixed = False
    times = math.ceil(len(data_in_out_name_shape)/len(model_in_out_name_shape))
    new_linenos, new_lines = [], []
    for d, m, in zip(data_in_out_name_shape, model_in_out_name_shape * times):
        data_shape = d.shape
        model_shape = m.shape
        if model_shape[1:] == data_shape:  # data shape lacks the batch size dimension
            var_name, lineno = d.var.name, d.var.lineno
            if d.replace:
                source[except_lineno - 1] = \
                    source[except_lineno - 1].replace(d.var.name, f'np.expand_dims({var_name}, 0)')
            else:
                blank = calculate_blank(source, lineno)
                line = blank + f'{var_name} = np.expand_dims({var_name}, 0)\n'
                new_linenos.append(lineno + 1)
                new_lines.append(line)
            fixed = True
    source = insert_new_lines(source, new_linenos, new_lines)
    return fixed, source

def expand_tf_x_data_last_dim(source, except_lineno, data_in_out_name_shape, model_in_out_name_shape):
    fixed = False
    times = math.ceil(len(data_in_out_name_shape)/len(model_in_out_name_shape))
    new_linenos, new_lines = [], []
    for d, m, in zip(data_in_out_name_shape, model_in_out_name_shape * times):
        data_shape, model_shape = d.shape, m.shape
        if data_shape == model_shape[:-1]  and model_shape[-1] == 1:  # data shape lacks the last dimension
            var_name, lineno = d.var.name, d.var.lineno
            if d.replace:
                source[except_lineno - 1] = \
                    source[except_lineno - 1].replace(d.var.name, f'np.expand_dims({var_name}, -1)')
            else:
                blank = calculate_blank(source, lineno)
                line = blank + f'{var_name} = np.expand_dims({var_name}, -1)\n'
                new_linenos.append(lineno + 1)
                new_lines.append(line)
            fixed = True
    source = insert_new_lines(source, new_linenos, new_lines)
    return fixed, source

def reshape_tf_x_data(source, except_lineno, data_in_out_name_shape, model_in_out_name_shape):
    fixed = False
    new_linenos, new_lines = [], []
    for i, d in enumerate(data_in_out_name_shape):
        x_data_shape = d.shape
        model_in_shape = model_in_out_name_shape[0].shape
        if len(x_data_shape) != 3:
            continue
        if i % 2 == 0 and x_data_shape[1]*x_data_shape[2] == model_in_shape[1]:
            var_name, lineno = d.var.name, d.var.lineno
            if d.replace:
                source[except_lineno - 1] = source[except_lineno - 1].replace(
                    d.var.name, f'{var_name}.reshape(-1, {model_in_shape[1]})')
            else:
                blank = calculate_blank(source, lineno)
                line = blank + f'{var_name} = {var_name}.reshape(-1, {model_in_shape[1]})\n'
                new_linenos.append(lineno + 1)
                new_lines.append(line)
            fixed = True
    source = insert_new_lines(source, new_linenos, new_lines)
    return fixed, source

def reshape_tf_y_data(source, except_lineno, data_in_out_name_shape):
    fixed = False
    new_linenos, new_lines = [], []
    for i, d in enumerate(data_in_out_name_shape):
        if i % 2 == 1 and len(d.shape) == 3 and d.shape[1] == 1:
            var_name, lineno = d.var.name, d.var.lineno
            if d.replace:
                source[except_lineno - 1] = source[except_lineno - 1].replace(
                    d.var.name, f'{var_name}.reshape(-1, {d.shape[-1]})')
            else:
                blank = calculate_blank(source, lineno)
                line = blank + f'{var_name} = {var_name}.reshape(-1, {d.shape[-1]})\n'
                new_linenos.append(lineno + 1)
                new_lines.append(line)
            fixed = True
    source = insert_new_lines(source, new_linenos, new_lines)
    return fixed, source

def modify_tf_model_input_shape(source, visitor, data_in_out_name_shape, model_in_out_name_shape):
    fixed = False
    if len(data_in_out_name_shape) > 0:
        x_data_shape = data_in_out_name_shape[0].shape[1:]
        model_input_shape = model_in_out_name_shape[0].shape[1:]
        if len(data_in_out_name_shape[0].shape) != 1 and model_input_shape != x_data_shape:
            model_input_var = model_in_out_name_shape[0].var
            model_input_lineno = model_input_var.lineno
            model_input_func = visitor.lineno_function_call[model_input_lineno][0]

            _, raw_param = visitor.func_key_raw_params[model_input_func][1]
            raw_input_shape, lineno = raw_param.name, raw_param.start_lineno
            source[lineno - 1] = source[lineno - 1].replace(raw_input_shape, str([None]+x_data_shape))
            fixed = True
    return fixed, source

def modify_tf_model_output_shape(source, visitor, data_in_out_name_shape, model_in_out_name_shape):
    fixed = False
    if len(data_in_out_name_shape) > 0:
        y_data_shape = data_in_out_name_shape[1].shape[1:]
        model_output_shape = model_in_out_name_shape[1].shape[1:]
        if model_output_shape != y_data_shape:
            model_output_var = model_in_out_name_shape[1].var
            model_output_lineno = model_output_var.lineno
            model_output_func = visitor.lineno_function_call[model_output_lineno][0]

            _, raw_param = visitor.func_key_raw_params[model_output_func][1]
            raw_output_shape, lineno = raw_param.name, raw_param.start_lineno
            source[lineno - 1] = source[lineno - 1].replace(raw_output_shape, str([None]+y_data_shape))
            fixed = True
    return fixed, source

def expand_keras_x_data_last_dim(source, first_layer_name, data_in_out_name_shape, model_in_out_name_shape):
    fixed = False
    model_in_shape = model_in_out_name_shape[0].shape
    if 'Conv1D' in first_layer_name or 'LSTM' in first_layer_name \
            or 'Conv2D' in first_layer_name or 'Convolution2D' in first_layer_name:
        new_linenos, new_lines = [], []
        for i, d in enumerate(data_in_out_name_shape):
            if i % 2 == 0 and d.shape[1:] == model_in_shape[1:-1]:
                var_name, lineno = d.var.name, d.var.lineno
                if d.replace:
                    source[lineno - 1] = source[lineno - 1].replace(var_name, f'np.expand_dims({var_name}, -1)')
                else:
                    blank = calculate_blank(source, lineno)
                    line = blank + f'{var_name} = np.expand_dims({var_name}, -1)\n'
                    new_linenos.append(lineno + 1)
                    new_lines.append(line)
                fixed = True
        source = insert_new_lines(source, new_linenos, new_lines)

    return fixed, source

def expend_SimpleRNN_data(source, first_layer_name, data_in_out_name_shape):
    fixed = False
    if 'SimpleRNN' in first_layer_name:
        new_linenos, new_lines = [], []
        for i, d in enumerate(data_in_out_name_shape):
            if i % 2 == 0 and len(d.shape) == 2:
                var_name, lineno = d.var.name, d.var.lineno
                if d.replace:
                    source[lineno - 1] = source[lineno - 1].replace(var_name, f'np.expand_dims({var_name}, 0)')
                else:
                    blank = calculate_blank(source, lineno)
                    line = blank + f'{var_name} = np.expand_dims({var_name}, 0)\n'
                    new_linenos.append(lineno + 1)
                    new_lines.append(line)
                fixed = True
        source = insert_new_lines(source, new_linenos, new_lines)
    return fixed, source

def transpose_x_data(source, except_lineno, data_in_out_name_shape, model_in_out_name_shape):
    fixed = False
    new_linenos, new_lines = [], []
    for i, d in enumerate(data_in_out_name_shape):
        x_data_shape = d.shape
        model_in_shape = model_in_out_name_shape[0].shape
        if i % 2 == 0 and x_data_shape[0] == model_in_shape[1] and x_data_shape[1] == model_in_shape[0]:
            var_name, lineno = d.var.name, d.var.lineno
            if d.replace:
                source[except_lineno - 1] = source[except_lineno - 1].replace(
                    d.var.name, f'{var_name}.transpose()')
            else:
                blank = calculate_blank(source, lineno)
                line = blank + f'{var_name} = {var_name}.transpose()\n'
                new_linenos.append(lineno + 1)
                new_lines.append(line)
            fixed = True
    source = insert_new_lines(source, new_linenos, new_lines)
    return fixed, source

def modify_y_data_shape(source, loss_function, data_in_out_name_shape, model_in_out_name_shape, replace=False):
    fixed = False
    if hasattr(loss_function, '__name__'):
        loss_function = loss_function.__name__
    if loss_function in re_loss_function or loss_function in {'binary_crossentropy', 'sparse_categorical_crossentropy'}:
        new_linenos, new_lines = [], []
        for i, d in enumerate(data_in_out_name_shape):
            if i % 2 == 1 and len(d.shape) == 1:
                var_name, lineno = d.var.name, d.var.lineno
                if d.replace:
                    source[lineno - 1] = source[lineno - 1].replace(var_name, f'np.expand_dims({var_name}, -1)')
                else:
                    blank = calculate_blank(source, lineno)
                    line = blank + f'{var_name} = np.expand_dims({var_name}, -1)\n'
                    new_linenos.append(lineno+1)
                    new_lines.append(line)
                fixed = True
        source = insert_new_lines(source, new_linenos, new_lines)

    if loss_function == 'categorical_crossentropy':
        class_num = model_in_out_name_shape[1].shape[-1]
        new_linenos, new_lines = [], []
        for i, d in enumerate(data_in_out_name_shape):
            if i % 2 == 1 and len(d.shape) == 1:
                var_name, lineno = d.var.name, d.var.lineno
                if d.replace:
                    source[lineno - 1] = source[lineno - 1].replace(var_name, f'np.eye({class_num})[{var_name}]')
                else:
                    blank = calculate_blank(source, lineno)
                    line = blank + f'{var_name} = np.eye({class_num})[{var_name}]\n'
                    new_linenos.append(lineno+1)
                    new_lines.append(line)
                fixed = True
        source = insert_new_lines(source, new_linenos, new_lines)

    if loss_function == 'sparse_categorical_crossentropy':
        new_linenos, new_lines = [], []
        for i, d in enumerate(data_in_out_name_shape):
            if i % 2 == 1 and d.shape[1] > 1:
                var_name, lineno = d.var.name, d.var.lineno
                if d.replace:
                    source[lineno - 1] = source[lineno - 1].replace(var_name, f'np.argmax({var_name}, axis=-1)')
                else:
                    blank = calculate_blank(source, lineno)
                    line = blank + f'{var_name} = np.argmax({var_name}, axis=-1)\n'
                    new_linenos.append(lineno+1)
                    new_lines.append(line)
                fixed = True
        source = insert_new_lines(source, new_linenos, new_lines)

    return fixed, source

def modify_model_input_shape(source, lineno, raw_input_shape,
                             model_input_shape, data_in_out_name_shape, param_type):
    fixed = False
    if param_type == 'input_shape':
        search_result = re.search(r'.*\.shape', raw_input_shape)
        if search_result:
            shape_param = search_result.group()
            source[lineno - 1] = source[lineno - 1].replace(raw_input_shape, shape_param+'[1:]')
            fixed = True
            return fixed, source

    if len(data_in_out_name_shape) > 0:
        x_data_shape = data_in_out_name_shape[0].shape[1:]
        if len(data_in_out_name_shape[0].shape) != 1 and model_input_shape != x_data_shape:
            if param_type == 'input_dim':
                x_data_shape  = x_data_shape[0]
            source[lineno - 1] = source[lineno - 1].replace(raw_input_shape, str(x_data_shape))
            fixed = True
    return fixed, source

def modify_model_batch_input_shape(source, lineno, raw_input_shape,
                                   model_input_shape, data_in_out_name_shape):
    fixed = False
    if model_input_shape[1:] != data_in_out_name_shape[0].shape[1:]:
        source[lineno - 1] = source[lineno - 1].replace(raw_input_shape,
                                                              str([model_input_shape[0]]+
                                                                  data_in_out_name_shape[0].shape[1:]))
        fixed = True
    return fixed, source

def modify_model_output_shape(source, lineno, raw_output_shape,
                              loss_function, data_in_out_name_shape, model_in_out_name_shape):
    fixed = False
    model_output_shape = model_in_out_name_shape[1].shape
    if len(model_output_shape) == 2:
        model_output_class_num = model_output_shape[1]
    else:
        return fixed, source
    if loss_function in re_loss_function:
        x_data_shape = data_in_out_name_shape[0].shape
        if x_data_shape[1] == 1 and model_output_class_num != 1:
            source[lineno - 1] = source[lineno - 1].replace(raw_output_shape, '1')
            fixed = True
    if model_output_class_num != data_in_out_name_shape[0].shape[-1]:
        source[lineno - 1] = source[lineno - 1].replace(raw_output_shape, str(data_in_out_name_shape[1].shape[-1]))
        fixed = True
    return fixed, source

def add_flatten_layer(source, data_in_out_name_shape, model_in_out_name_shape):
    fixed = False
    y_data_dimension = len(data_in_out_name_shape[1].shape)
    model_out_shape_dimension = len(model_in_out_name_shape[1].shape)
    if y_data_dimension == 2 and model_out_shape_dimension > y_data_dimension:
        last_layer_lineno  = model_in_out_name_shape[1].var.lineno
        last_layer_source = source[last_layer_lineno-1]
        start = last_layer_source.find('Dense')
        end = find_parameter_end(source, last_layer_lineno, start)[1]
        flatten = last_layer_source.replace(last_layer_source[start:end], 'Flatten()')
        source.insert(last_layer_lineno - 1, flatten)
        fixed = True

    return fixed, source

def tf__nn__bidirectional_dynamic_rnn(source, function_key, visitor, context):
    fixed = False
    for key, raw_param in visitor.func_key_raw_params[function_key]:
        excep_lineno = raw_param.start_lineno
        raw_param = raw_param.name
        if key == 'sequence_length':
            real_param = eval(raw_param, context[0], context[1])
            real_param_shape = resolve_shape(real_param)
            if real_param_shape == [1]:
                source[excep_lineno-1] = source[excep_lineno-1].replace(raw_param, f'[{raw_param}]*batch_size')
                fixed = True
    return fixed, source

def tf__matmul(source, function_key, visitor, context):
    fixed = False
    for i, (key, raw_param) in enumerate(visitor.func_key_raw_params[function_key]):
        excep_lineno = raw_param.start_lineno
        raw_param = raw_param.name
        data_dependency = find_data_dependency(function_key, raw_param, visitor)
        if data_dependency:
            var = eval(data_dependency.name, context[0], context[1])
            var_shape = resolve_shape(var)
            var_lineno = data_dependency.lineno
            if len(var_shape) == 1:
                source = insert_or_replace(
                    source, raw_param, var_lineno, f'tf.expand_dims({raw_param}, {i})', 'insert')
                fixed = True
                break
        else:
            real_param = eval(raw_param, context[0], context[1])
            real_param_shape = resolve_shape(real_param)
            if len(real_param_shape) == 1:
                source = insert_or_replace(
                    source, raw_param, excep_lineno, f'tf.expand_dims({raw_param}, {i})', 'replace')
                fixed = True
                break
    return fixed, source

def tf__confusion_matrix(source, function_key, visitor, context):
    fixed = False
    for key, raw_param in visitor.func_key_raw_params[function_key]:
        excep_lineno = raw_param.start_lineno
        raw_param = raw_param.name
        if key == 'labels' or key == 'predictions':
            real_param = eval(raw_param, context[0], context[1])
            real_param_shape = resolve_shape(real_param)
            if len(real_param_shape) == 2:
                source[excep_lineno-1] = source[excep_lineno-1].replace(raw_param, f'tf.argmax({raw_param}, -1)')
                fixed = True
    return fixed, source

def tf__losses__sparse_softmax_cross_entropy(source, function_key, visitor, context):
    fixed = False
    for key, raw_param in visitor.func_key_raw_params[function_key]:
        excep_lineno = raw_param.start_lineno
        raw_param = raw_param.name
        if key == 'labels':
            data_dependency = find_data_dependency(function_key, raw_param, visitor)
            if data_dependency:
                var = eval(data_dependency.name, context[0], context[1])
                var_shape = resolve_shape(var)
                var_lineno = data_dependency.lineno
                if var_shape[1] > 1:
                    source = insert_or_replace(
                        source, raw_param, var_lineno, f'tf.argmax({raw_param}, -1)', 'insert')
                    fixed = True
            else:
                real_param = eval(raw_param, context[0], context[1])
                real_param_shape = resolve_shape(real_param)
                if real_param_shape[1] > 1:
                    source = insert_or_replace(
                        source, raw_param, excep_lineno, f'tf.argmax({raw_param}, -1)', 'replace')
                    fixed = True

    return fixed, source

def tf__image__non_max_suppression(source, function_key, visitor, context):
    fixed = False
    for key, raw_param in visitor.func_key_raw_params[function_key]:
        excep_lineno = raw_param.start_lineno
        raw_param = raw_param.name
        if key == 'boxes' or key == 'scores':
            data_dependency = find_data_dependency(function_key, raw_param, visitor)
            if data_dependency:
                var = eval(data_dependency.name, context[0], context[1])
                var_shape = resolve_shape(var)
                var_lineno = data_dependency.lineno
                if key == 'boxes' and len(var_shape) == 3 or key == 'scores' and len(var_shape) == 2 :
                    source = insert_or_replace(
                        source, raw_param, var_lineno, f'{raw_param}.squeeze(0)', 'insert')
                    fixed = True
                    break
            else:
                real_param = eval(raw_param, context[0], context[1])
                real_param_shape = resolve_shape(real_param)
                if key == 'boxes' and len(real_param_shape) == 3 or key == 'scores' and len(real_param_shape) == 2 :
                    source = insert_or_replace(
                        source, raw_param, excep_lineno, f'{raw_param}.squeeze(0)', 'replace')
                    fixed = True
                    break

    return fixed, source

def tf__nn__conv2d(source, function_key, visitor, context):
    fixed = False
    for i, (key, raw_param) in enumerate(visitor.func_key_raw_params[function_key]):
        excep_lineno = raw_param.start_lineno
        raw_param = raw_param.name
        if i == 0:
            data_dependency = find_data_dependency(function_key, raw_param, visitor)
            if data_dependency:
                var = eval(data_dependency.name, context[0], context[1])
                var_shape = resolve_shape(var)
                var_lineno = data_dependency.lineno
                if len(var_shape) == 3:
                    source = insert_or_replace(
                        source, raw_param, var_lineno, f'np.expand_dims({raw_param}, 0)', 'insert')
                    fixed = True
                    break
            else:
                real_param = eval(raw_param, context[0], context[1])
                real_param_shape = resolve_shape(real_param)
                if len(real_param_shape) == 3:
                    source = insert_or_replace(
                        source, raw_param, excep_lineno, f'np.expand_dims({raw_param}, 0)', 'replace')
                    fixed = True
                    break
    return fixed, source

def tf__sets__intersection(source, function_key, visitor, context):
    fixed = False
    for i, (key, raw_param) in enumerate(visitor.func_key_raw_params[function_key]):
        excep_lineno = raw_param.start_lineno
        raw_param = raw_param.name
        if i == 0 or i == 1:
            data_dependency = find_data_dependency(function_key, raw_param, visitor)
            if data_dependency:
                var = eval(data_dependency.name, context[0], context[1])
                var_shape = resolve_shape(var)
                var_lineno = data_dependency.lineno
                if len(var_shape) == 1:
                    source = insert_or_replace(
                        source, raw_param, var_lineno, f'tf.expand_dims({raw_param}, 0)', 'insert')
                    fixed = True
                    break
            else:
                real_param = eval(raw_param, context[0], context[1])
                real_param_shape = resolve_shape(real_param)
                if len(real_param_shape) == 1:
                    source = insert_or_replace(
                        source, raw_param, excep_lineno, f'tf.expand_dims({raw_param}, 0)', 'replace')
                    fixed = True
                    break
    return fixed, source

