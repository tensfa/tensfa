import os
import re
import pandas as pd

def test():
    for f in os.listdir('/mnt/sdb/wdw/TestData/ISSTA2018NotTensorShapeFault'):
        file_path = f'/mnt/sdb/wdw/TestData/ISSTA2018NotTensorShapeFault/{f}'
        command = f'/mnt/sdb/wdw/doop/doop -a 1-call-site-sensitive+heap -i {file_path} -id ut1 --platform python_2 ' \
                  '--single-file-analysis --tensor-shape-analysis --full-tensor-precision'
        os.system(command)

def merge():
    output = open('ISSTA2018NotTensorShapeFault_pythia_test_result.txt').read()
    question_ids = re.findall(r'/mnt/sdb/wdw/TestData/ISSTA2018NotTensorShapeFault/s([0-9]+)_context.py -id ut1 '
                              r'--platform python_2 --single-file-analysis '
                              r'--tensor-shape-analysis --full-tensor-precision', output)
    op_error_ins = re.findall(r'tensor op error \(INS\)\s*([0-9]+)\n', output)
    op_error_sens = re.findall(r'tensor op error \(SENS\)\s*([0-9]+)\n', output)
    op_warning_ins = re.findall(r'tensor op warning \(INS\)\s*([0-9]+)\n', output)
    op_warning_sens = re.findall(r'tensor op warning \(SENS\)\s*([0-9]+)\n', output)
    data1 = pd.DataFrame({'question id': question_ids,
                         'pythia: tensor op error (INS)': op_error_ins,
                         'pythia: tensor op error (SENS)': op_error_sens,
                         'pythia: tensor op warning (INS)': op_warning_ins,
                         'pythia: tensor op warning (SENS)': op_warning_sens})
    for i in data1.columns:
        data1.loc[:, i] = data1.loc[:, i].apply(int)
    data2 = pd.read_excel('../ISSTA2018NoTensorShapeFault.xlsx')
    data = pd.merge(data2, data1, on='question id')
    data.to_excel('../ISSTA2018NoTensorShapeFault.xlsx', index=False)

if __name__ == '__main__':
    merge()