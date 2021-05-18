import ast
# import astunparse
from collections import defaultdict, namedtuple, deque, OrderedDict
from utils import find_parameter_end

class Visitor(ast.NodeVisitor):
    def __init__(self, source_lines):
        self.source_lines = source_lines

        self.lineno_varname = OrderedDict()
        self.lineno_function_call = OrderedDict()

        self.var = namedtuple('variable', ['name', 'lineno', 'start_index'])
        # include start_index but not include end_index
        self.raw_param = namedtuple('raw_param', ['name', 'start_lineno', 'start_index', 'end_lineno', 'end_index'])
        self.func_key = namedtuple('func_key', ['name', 'lineno', 'start_index'])

        self.func_key_var_params = defaultdict(list) # func_key:[var or func_key]
        self.func_key_raw_params = defaultdict(list) # func_key:[(None or key, raw_param)]
        self.func_key_return_value = defaultdict(list) # func_key:[var]

        self.graph = defaultdict(list)
        self.store = [defaultdict(list)]
        self.load = [{}]

        self._current_called_function_key = []
        self._current_attributes = []

    def visit_FunctionDef(self, node):
        self.store.append(defaultdict(list))
        self.load.append({})
        for arg in node.args.args:
            self.store[-1][arg.arg].append(self.var(arg.arg, arg.lineno, arg.col_offset))
        super(Visitor, self).generic_visit(node)
        self.store.pop()
        self.load.pop()

    def visit_Attribute(self, node):
        self._current_attributes.append(node.attr)
        super(Visitor, self).generic_visit(node)
        self._current_attributes = []

    def visit_Name(self, node):
        if self._current_attributes:
            attribute = '.'.join(self._current_attributes[::-1])
            name = node.id + '.' + attribute
        else:
            name = node.id
        v = self.var(name, node.lineno, node.col_offset)
        if isinstance(node.ctx, ast.Store):
            self.store[-1][name].append(v)
            if node.lineno not in self.lineno_varname:
                self.lineno_varname[node.lineno] = []
            self.lineno_varname[node.lineno].append(name)
        elif isinstance(node.ctx, ast.Load):
            self.load[-1][name] = v
            self.create_edge_by_lineno(v)
            self.create_edge_by_name(v)
        elif isinstance(node.ctx, ast.Del):
            pass
        elif isinstance(node.ctx, ast.AugLoad):
            pass
        elif isinstance(node.ctx, ast.AugStore):
            pass
        elif isinstance(node.ctx, ast.Param):
            self.load[-1][name] = v

        if self._current_called_function_key and self._current_called_function_key[-1] != v:
            self.func_key_var_params[self._current_called_function_key[-1]].append(v)

    def create_edge_by_lineno(self, load):
        for stores in self.store[-1].values():
            for store in stores:
                if store.lineno == load.lineno:
                    self.graph[store].append(load)

    def create_edge_by_name(self, load):
        for store_dict in self.store[::-1]:
            for name, stores in store_dict.items():
                for store in stores:
                    if store.lineno != load.lineno and store.name == load.name:
                        self.graph[load].append(store)

    def query_data_dependency(self, variable):
        visited = [variable]
        children = deque(self.graph[variable])
        if not children:
            return []

        while children:
            variable = children.popleft()
            if variable not in visited:
                children.extend(deque(self.graph[variable]))
                visited.append(variable)

        return visited

    def build_control_dependency(self): # TODO
        pass

    def visit_Assign(self, node):
        targets = []
        for t in node.targets:
            if isinstance(t, ast.Name):
                targets.append(self.var(t.id, t.lineno, t.col_offset))
            elif isinstance(t, ast.Tuple):
                for tt in t.elts:
                    if isinstance(tt, ast.Name):
                        targets.append(self.var(tt.id, tt.lineno, tt.col_offset))
                    elif isinstance(tt, ast.Tuple):
                        for ttt in tt.elts:
                            if isinstance(ttt, ast.Name):
                                targets.append(self.var(ttt.id, ttt.lineno, ttt.col_offset))
                            else:
                                print('NotImplementedError')
                    else:
                        print('NotImplementedError')
            else:
                print('NotImplementedError')

        for t in node.targets:
            super(Visitor, self).visit(t)

        if isinstance(node.value, ast.Call):
            function_key = self.visit_Call(node.value)
            self.func_key_return_value[function_key] = targets
        else:
            super(Visitor, self).visit(node.value)

    def parse_raw_param(self, start_lineno, start_index):
        end_lineno, end_index = find_parameter_end(self.source_lines, start_lineno, start_index)
        if start_lineno == end_lineno:
            raw_param_name = self.source_lines[start_lineno - 1][start_index:end_index]
        else:
            raw_param_name = self.source_lines[start_lineno - 1][start_index:]
            i = start_lineno
            while i < end_lineno - 1:
                raw_param_name += self.source_lines[i]
                i += 1
            raw_param_name += self.source_lines[end_lineno - 1][:end_index]
        raw_param = self.raw_param(raw_param_name, start_lineno, start_index, end_lineno, end_index)
        return raw_param

    def visit_Call(self, node):
        if hasattr(node.func, 'func'):
            return self.visit_Call(node.func)
        elif not hasattr(node.func, 'value'): # a()
            function_name = node.func.id
        elif isinstance(node.func.value, ast.Name): # a.b()
            function_name = '%s.%s' % (node.func.value.id, node.func.attr)
        elif isinstance(node.func.value, ast.Attribute): # a.b.c()
            attrs = []
            value = node.func.value
            while isinstance(value, ast.Attribute):
                attrs.append(value.attr)
                value = value.value
            attr = '.'.join(attrs[::-1])
            function_name = '%s.%s.%s' % (value.id, attr, node.func.attr)
        elif isinstance(node.func.value, ast.Call):
            function_name = '[call]'
        else:
            function_name = '[unknown]'
        function_key = self.func_key(function_name, node.lineno, node.col_offset)
        if node.lineno not in self.lineno_function_call:
            self.lineno_function_call[node.lineno] = []
        self.lineno_function_call[node.lineno].append(function_key)
        self._current_called_function_key.append(function_key)

        for arg in node.args:
            # raw_param_name = astunparse.unparse(arg).strip()
            start_index = arg.col_offset
            if isinstance(arg, ast.Tuple):
                start_index -= 1
            raw_param = self.parse_raw_param(arg.lineno, start_index)
            self.func_key_raw_params[function_key].append((None, raw_param))
        for keyword in node.keywords:
            karg, kvalue = keyword.arg, keyword.value
            start_index = kvalue.col_offset
            if isinstance(kvalue, ast.Tuple):
                start_index -= 1
            raw_param = self.parse_raw_param(kvalue.lineno, start_index)
            self.func_key_raw_params[function_key].append((karg, raw_param))

        super(Visitor, self).generic_visit(node)
        last = self._current_called_function_key.pop()
        if self._current_called_function_key:
            self.func_key_var_params[self._current_called_function_key[-1]].append(last)
        return function_key