import importlib
import pandas as pd
import threading
from tqdm import tqdm
import glob

def test():
    files = glob.glob('TestData/ISSTA2018NotTensorShapeFault/s[0-9]*_context.py')
    question_ids = [f.lstrip('TestData/ISSTA2018NotTensorShapeFault/s').rstrip('_context.py') for f in files]

    exception_ids, exception_types, exception_strings=[], [], []
    def collect(i):
        try:
            module = importlib.import_module(f'TestData.ISSTA2018NotTensorShapeFault.s{i}_context')
            if hasattr(module, 'main'):
                module.main()
            exception_ids.append(i)
            exception_types.append(None)
            exception_strings.append(None)
        except Exception as e:
            exception_type = type(e).__name__
            exception_s = str(e)
            exception_ids.append(i)
            exception_types.append(exception_type)
            exception_strings.append(exception_s)
    for qi in tqdm(question_ids):
        t = threading.Thread(target=collect, args=(qi,))
        t.start()
        t.join()
    exception_data = pd.DataFrame({'question id': exception_ids,
                             'ShapeFlow: exception type': exception_types,
                             'ShapeFlow: exception content': exception_strings})
    exception_data['question id'] = exception_data['question id'].apply(int)
    exception_data.to_excel('ISSTA2018NotTensorShapeFault_ShapeFlow_test_result.xlsx', index=False)

def merge():
    data1 = pd.read_excel('ISSTA2018NotTensorShapeFault_ShapeFlow_test_result.xlsx')
    data2 = pd.read_excel('../ISSTA2018NoTensorShapeFault.xlsx')
    data = pd.merge(data2, data1, on='question id')
    data.to_excel('../ISSTA2018NoTensorShapeFault.xlsx', index=False)

if __name__ == '__main__':
    merge()