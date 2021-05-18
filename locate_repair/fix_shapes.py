import ast
import importlib
import sys
import traceback
import threading
import os

from collections import namedtuple
from tqdm import tqdm

import collect_shapes
import fix_patterns
import pandas as pd

from ast_parser import Visitor
from utils import resolve_shape, split_dict_string, split_tuple_or_list_string
sys.path.append('../')
from detect.predict import predict

class ShapeFixer:
    
    def __init__(self, data_dir):
        self.first_fix = True
        self.fixed = False
        self.data_dir = data_dir
        self.data_name_shape = namedtuple('data_name_shape', ['var', 'shape', 'replace'])
        self.model_name_shape = namedtuple('model_name_shape', ['var', 'shape'])

    def write_new_source_code(self, source_code, question_id, try_index):
        new_raw_source = ''.join(source_code)
        with open(f'{self.data_dir}/s{question_id}_repaired_{try_index}.py', 'w', encoding='utf-8') as f:
            f.write(new_raw_source)

    def parse_tf_data_model(self, except_function, visitor, source, f_globals, f_locals,
                            data_in_out_name_shape, model_in_out_name_shape):
        raw_parameters = visitor.func_key_raw_params[except_function]
        for rp in raw_parameters:
            if rp[0] == 'feed_dict' or rp[1].name.strip().startswith('{'):
                raw_feed_dict = rp[1]
                feed_dict = split_dict_string(raw_feed_dict.name)
                if not feed_dict:
                    feed_dict_var = visitor.var(rp[1].name,
                                                rp[1].start_lineno, rp[1].start_index)
                    if feed_dict_var not in visitor.graph:
                        break
                    feed_dict_var = visitor.graph[feed_dict_var][0]
                    if feed_dict_var not in visitor.graph:
                        break
                    data_dependency = visitor.graph[feed_dict_var]
                    data_dependency = sorted(data_dependency, key=lambda d: d.start_index)
                    for i in [0, 2]:
                        k_var, v_var = data_dependency[i], data_dependency[i + 1]
                        k, v = k_var.name, v_var.name

                        real_k = eval(k, f_globals, f_locals)
                        real_v = eval(v, f_globals, f_locals)
                        real_k_shape = resolve_shape(real_k)  # model related shape
                        real_v_shape = resolve_shape(real_v)  # data related shape

                        if k_var in visitor.graph:
                            k_var = visitor.graph[k_var][-1]
                            real_k_shape = resolve_shape(eval(k_var.name, f_globals, f_locals))
                        k_data = self.model_name_shape(k_var, real_k_shape)
                        if k_data not in model_in_out_name_shape:
                            model_in_out_name_shape.append(k_data)

                        if v_var in visitor.graph:
                            v_var = visitor.graph[v_var][-1]
                            real_v_shape = resolve_shape(eval(v_var.name, f_globals, f_locals))
                            replace = False
                        else:
                            replace = True
                        v_data = self.data_name_shape(v_var, real_v_shape, replace)
                        if v_data not in data_in_out_name_shape:
                            data_in_out_name_shape.append(v_data)

                for k, v in feed_dict.items():
                    real_k = eval(k, f_globals, f_locals)
                    real_v = eval(v, f_globals, f_locals)
                    if isinstance(real_v, float):
                        continue
                    real_k_shape = resolve_shape(real_k)  # model related shape
                    real_v_shape = resolve_shape(real_v)  # data related shape
                    if real_k_shape == [1]:
                        continue

                    k_var, v_var = None, None
                    finded_k_var, finded_v_var = False,  False
                    for var in visitor.func_key_var_params[except_function]:
                        if var.name == k:
                            k_var = var
                            finded_k_var = True
                        elif var.name == v:
                            v_var = var
                            finded_v_var = True

                    start_lineno, end_lineno = raw_feed_dict.start_lineno, raw_feed_dict.end_lineno
                    kline, vline, ki, vi = -1, -1, -1, -1
                    if not finded_k_var:
                        for i in range(start_lineno, end_lineno + 1):
                            line = source[i - 1]
                            ki = line.find(k)
                            if ki != -1:
                                kline = i
                                break
                        k_var = visitor.var(k, kline, ki)

                    if not finded_v_var:
                        for i in range(start_lineno, end_lineno + 1):
                            line = source[i - 1]
                            vi = line.find(v)
                            if vi != -1:
                                vline = i
                                break
                        v_var = visitor.var(v, vline, vi)

                    if k_var in visitor.graph:
                        k_var = visitor.graph[k_var][-1]
                        real_k_shape = resolve_shape(eval(k_var.name, f_globals, f_locals))
                    k_data = self.model_name_shape(k_var, real_k_shape)
                    if k_data not in model_in_out_name_shape:
                        model_in_out_name_shape.append(k_data)

                    if v_var in visitor.graph:
                        v_var = visitor.graph[v_var][-1]
                        real_v_shape = resolve_shape(eval(v_var.name, f_globals, f_locals))
                        replace = False
                    else:
                        replace = True
                    v_data = self.data_name_shape(v_var, real_v_shape, replace)
                    if v_data not in data_in_out_name_shape:
                        data_in_out_name_shape.append(v_data)
                break
        return data_in_out_name_shape, model_in_out_name_shape


    @staticmethod
    def parse_tf_loss_function(visitor):
        loss_function = 'categorical_crossentropy'
        finded_loss_function = False
        for _, functions in visitor.lineno_function_call.items():
            for func in functions:
                if 'sparse_softmax_cross_entropy_with_logits' in func.name:
                    loss_function = 'sparse_categorical_crossentropy'
                    finded_loss_function = True
                    break
                elif 'softmax_cross_entropy_with_logits' in func.name:
                    loss_function = 'categorical_crossentropy'
                    finded_loss_function = True
                    break

            if finded_loss_function:
                break
        return loss_function

    @staticmethod
    def parse_keras_first_layer_function(visitor, f_globals, f_locals):
        first_layer_function = None
        finded_first_layer = False
        for _, funcs in visitor.lineno_function_call.items():
            for f in funcs:
                try:
                    var = eval(f.name, f_globals, f_locals)
                    if var.__name__ == 'Input':
                        first_layer_function = f
                        finded_first_layer = True
                        break
                    if hasattr(var, '__mro__'):
                        mro = {c.__name__ for c in var.__mro__}
                        if 'Model' not in mro and 'Layer' in mro:
                            first_layer_function = f
                            finded_first_layer = True
                            break
                except Exception as e:
                    print(e)
            if finded_first_layer:
                break
        return first_layer_function

    def parse_kera_data(self, visitor, source, f_globals, f_locals, data_in_out_name_shape):
        for _, funcs in visitor.lineno_function_call.items():
            for f in funcs:
                splited_f = f.name.split('.')
                if splited_f[-1] in {'fit', 'evaluate', 'predict', 'train'}:
                    try:
                        var = eval(''.join(splited_f[:-1]), f_globals, f_locals)
                        if splited_f[-1] == 'train':
                            f = visitor.func_key_var_params[f][0]
                        if var.__class__.__name__ in ['Model', 'Sequential', 'Estimator']:
                            data_params = []
                            for i, (key, raw_param) in enumerate(visitor.func_key_raw_params[f]):
                                if i < 2 and key in [None, 'x', 'y']:
                                    param = visitor.var(raw_param.name.strip(),
                                                        raw_param.start_lineno,
                                                        raw_param.start_index)
                                    data_params.append(param)
                                elif key == 'validation_data':
                                    params = split_tuple_or_list_string(raw_param.name)
                                    for param in params:
                                        i = raw_param.start_lineno - 1
                                        while i < raw_param.end_lineno:
                                            param_index = source[i].find(param)
                                            if param_index != -1:
                                                var_param = visitor.var(
                                                    param,raw_param.start_lineno,param_index)
                                                data_params.append(var_param)
                                                break
                                            i += 1

                            for param in data_params:
                                if param in visitor.graph:
                                    data_dependency = visitor.graph[param]
                                    real_param = eval(data_dependency[-1].name, f_globals, f_locals)
                                    real_param_shape = resolve_shape(real_param)
                                    if (data_dependency[-1],
                                        real_param_shape) not in data_in_out_name_shape:
                                        d = self.data_name_shape(data_dependency[-1], real_param_shape, False)
                                        data_in_out_name_shape.append(d)
                                else:
                                    real_param = eval(param.name, f_globals, f_locals)
                                    real_param_shape = resolve_shape(real_param)
                                    d = self.data_name_shape(param, real_param_shape, True)
                                    data_in_out_name_shape.append(d)
                    except Exception as e:
                        print(e)

    @staticmethod
    def parse_keras_model_var(visitor, f_globals, f_locals):
        model = None
        finded_model = False
        for _, vs in visitor.lineno_varname.items():
            for v in vs:
                try:
                    real_v = eval(v, f_globals, f_locals)
                    if hasattr(real_v, 'layers') and hasattr(real_v, 'loss'):
                        model = real_v
                        finded_model = True
                        break
                except Exception as e:
                    print(e)
            if finded_model:
                break
        return model

    def parse_keras_model(self, model, model_in_out_name_shape, visitor):
        input_layer_fname = model.layers[0].__class__.__name__
        output_layer_fname = 'Dense'
        input_layer_function_call = '[unknown]'
        output_layer_function_call = '[unknown]'
        finded_input_layer = False
        for _, function_calls in visitor.lineno_function_call.items():
            for function_call in function_calls:
                if input_layer_fname in function_call.name:
                    input_layer_function_call = function_call
                    finded_input_layer = True
                    break
            if finded_input_layer:
                break
        finded_output_layer = False
        for _, function_calls in list(visitor.lineno_function_call.items())[::-1]:
            for function_call in function_calls:
                if output_layer_fname in function_call.name:
                    output_layer_function_call = function_call
                    finded_output_layer = True
                    break
            if finded_output_layer:
                break
        model_in_name_shape = self.model_name_shape(input_layer_function_call, list(model.input_shape))
        if model_in_name_shape not in model_in_out_name_shape:
            model_in_out_name_shape.append(model_in_name_shape)
        model_out_name_shape = self.model_name_shape(output_layer_function_call, list(model.output_shape))
        if model_out_name_shape not in model_in_out_name_shape:
            model_in_out_name_shape.append(model_out_name_shape)

    def _fix_shape_incompatibility(self, question_id, try_index):
        self.fixed = False

        if self.first_fix:
            file_name = f's{question_id}_context'
            source = open(f'{self.data_dir}/{file_name}.py', 'r').readlines()
            self.write_new_source_code(source, question_id, 0)
            self.first_fix = False
        else:
            file_name = f's{question_id}_repaired_{try_index-1}'
            source = open(f'{self.data_dir}/{file_name}.py', 'r').readlines()
        raw_source = ''.join(source)
        if 'keras' in raw_source:
            tensorflow_or_keras = 'keras'
        elif 'tensorflow' in raw_source:
            tensorflow_or_keras = 'tensorflow'
        else:
            return
        root_node = ast.parse(raw_source, file_name)
        visitor = Visitor(source)
        visitor.visit(root_node)
    
        except_lineno = -1
        shape_error = False
        collect_shapes.init_shapes_collection(filenames=[f'{self.data_dir}\\{file_name}.py'],
                                              lineno_varname=visitor.lineno_varname)
        with collect_shapes.collect():
            try:
                module = importlib.import_module(f'{file_name}')
                if hasattr(module, 'main'):
                    module.main()
            except Exception as e:
                s = str(e)
                et, ev, tb = sys.exc_info()
                te = traceback.TracebackException(et, ev, tb)
                for s in te.stack:
                    if file_name in s.filename:
                        except_lineno = s.lineno
                        print(except_lineno)
                # if True:
                if predict(et.__name__ + ' ' + str(ev)):
                    print('tensor shape fault')
                    shape_error = True
                # fix：
                if shape_error:
                    data_in_out_name_shape = []
                    model_in_out_name_shape = []

                    # all_shape_info = collect_shapes.dumps_stats()
                    f_globals, f_locals = collect_shapes.get_context()
                    context = f_globals, f_locals

                    except_function_keys = None
                    tlineo = except_lineno
                    while tlineo >= 0:
                        if tlineo in visitor.lineno_function_call:
                            except_function_keys = visitor.lineno_function_call[tlineo]
                            break
                        tlineo -= 1
    
                    if tensorflow_or_keras == 'tensorflow':
                        loss_function = self.parse_tf_loss_function(visitor)

                        for except_func in except_function_keys:
                            if except_func not in visitor.func_key_var_params:
                                continue
                            splited_func = except_func.name.split('.')
                            if splited_func[-1] in {'run', 'eval'}:
                                self.parse_tf_data_model(except_func, visitor, source, f_globals, f_locals,
                                                         data_in_out_name_shape, model_in_out_name_shape)
                                if len(data_in_out_name_shape) == 0 or len(model_in_out_name_shape) == 0:
                                    continue
                                if not self.fixed:
                                    self.fixed, source = fix_patterns.expand_x_data_first_dim(
                                        source, except_lineno, data_in_out_name_shape, model_in_out_name_shape)
                                if not self.fixed:
                                    self.fixed, source = fix_patterns.expand_tf_x_data_last_dim(
                                        source, except_lineno, data_in_out_name_shape, model_in_out_name_shape)
                                if not self.fixed:
                                    self.fixed, source = fix_patterns.reshape_tf_x_data(
                                        source, except_lineno, data_in_out_name_shape, model_in_out_name_shape)
                                if not self.fixed:
                                    self.fixed, source = fix_patterns.modify_y_data_shape(
                                        source, loss_function,data_in_out_name_shape,model_in_out_name_shape)
                                if not self.fixed:
                                    self.fixed, source = fix_patterns.reshape_tf_y_data(
                                        source, except_lineno, data_in_out_name_shape)
                                if not self.fixed:
                                    self.fixed, source = fix_patterns.modify_tf_model_input_shape(
                                        source, visitor, data_in_out_name_shape, model_in_out_name_shape)
                                if not self.fixed:
                                    self.fixed, source = fix_patterns.modify_tf_model_output_shape(
                                        source, visitor, data_in_out_name_shape, model_in_out_name_shape)
    
                            function_name = except_func.name.replace('.', '__')
                            if hasattr(fix_patterns, function_name):
                                fix_function = getattr(fix_patterns, function_name)
                                self.fixed, source = fix_function(
                                    source, except_func, visitor, context)
                    else:
                        first_layer_function = self.parse_keras_first_layer_function(visitor, f_globals, f_locals)
                        self.parse_kera_data(visitor, source, f_globals, f_locals, data_in_out_name_shape)
                        model = self.parse_keras_model_var(visitor, f_globals, f_locals)
                        if hasattr(model, 'loss'):
                            loss_function = model.loss
                        else:
                            loss_function = '[unknown]'
                        if hasattr(model, 'input_shape') and hasattr(model, 'output_shape'):
                            self.parse_keras_model(model, model_in_out_name_shape, visitor)
                        else:
                            if not self.fixed:
                                self.fixed, source = fix_patterns.expend_SimpleRNN_data(
                                    source, first_layer_function.name, data_in_out_name_shape)

                        # check input layer
                        for key, rp in visitor.func_key_raw_params[first_layer_function]:
                            if key == 'input_shape' or key == 'input_dim':
                                real_param = eval(rp.name, f_globals, f_locals)
                                if isinstance(real_param, int):
                                    model_input_shape = [real_param]
                                else:
                                    model_input_shape = list(real_param)
                                if not self.fixed and model_in_out_name_shape:
                                    self.fixed, source = fix_patterns.expand_keras_x_data_last_dim(
                                        source,first_layer_function.name,data_in_out_name_shape, model_in_out_name_shape)
                                if not self.fixed:
                                    self.fixed, source = fix_patterns.modify_model_input_shape(
                                        source,rp.start_lineno, rp.name,
                                        model_input_shape,data_in_out_name_shape, key)
                            if key == 'batch_shape':
                                real_param = eval(rp.name, f_globals, f_locals)
                                model_input_shape = list(real_param)
                                if not self.fixed:
                                    self.fixed, source = fix_patterns.modify_model_batch_input_shape(
                                        source,rp.lineno, rp.name,
                                        model_input_shape,data_in_out_name_shape)

                        if self.fixed:
                            self.write_new_source_code(source, question_id, try_index)
                        if not model_in_out_name_shape:
                            return

                        for except_func in except_function_keys:
                            if except_func not in visitor.func_key_var_params:
                                continue
                            splited_func = except_func.name.split('.')
                            if splited_func[-1] in {'fit', 'evaluate', 'predict', 'train'}:
                                if not self.fixed:
                                    self.fixed, source = fix_patterns.expand_x_data_first_dim(
                                        source, except_lineno,data_in_out_name_shape, model_in_out_name_shape)
                                if not self.fixed:
                                    self.fixed, source = fix_patterns.transpose_x_data(
                                        source, except_lineno, data_in_out_name_shape, model_in_out_name_shape)
                                if not self.fixed:
                                    self.fixed, source = fix_patterns.modify_y_data_shape(
                                        source, loss_function, data_in_out_name_shape, model_in_out_name_shape)
                                if not self.fixed:
                                    for key, rp in visitor.func_key_raw_params[first_layer_function]:
                                        if key == 'input_shape' or key == 'input_dim':
                                            real_param = eval(rp.name, f_globals, f_locals)
                                            if isinstance(real_param, int):
                                                model_input_shape = [real_param]
                                            else:
                                                model_input_shape = list(real_param)
                                            self.fixed, source = fix_patterns.modify_model_input_shape(
                                                source,rp.start_lineno,rp.name,model_input_shape,
                                                data_in_out_name_shape,key)
                                if not self.fixed and model_in_out_name_shape:
                                    output_layer_function_call = model_in_out_name_shape[1].var
                                    raw_params = visitor.func_key_raw_params[output_layer_function_call]
                                    raw_output_shape = raw_params[0][1].name
                                    lineno = raw_params[0][1].start_lineno
                                    self.fixed, source = fix_patterns.modify_model_output_shape(
                                        source, lineno, raw_output_shape, loss_function,
                                        data_in_out_name_shape, model_in_out_name_shape)
                                if not self.fixed:
                                    self.fixed, source = fix_patterns.add_flatten_layer(
                                        source, data_in_out_name_shape, model_in_out_name_shape)
                    if self.fixed:
                        self.write_new_source_code(source, question_id, try_index)

    def fix_shape_incompatibility(self, question_id, max_try_number):
        for i in range(max_try_number):
            t = threading.Thread(target=self._fix_shape_incompatibility, args=(question_id, i))
            t.start()
            t.join()
            if not self.fixed:
                break

if __name__ == '__main__':
    data_source = sys.argv[1] # StackOverflow or ICSE2020ToRepair
    data_dir = os.path.dirname(os.path.realpath(__file__))+f'/../SFData/{data_source}'
    sys.path.append(data_dir)
    question_ids = pd.read_excel(os.path.dirname(os.path.realpath(__file__))+f'/../SFData/{data_source}.xlsx')['question id']
    for qi in tqdm(question_ids):
        shapeFixer = ShapeFixer(data_dir)
        shapeFixer.fix_shape_incompatibility(qi, 5)