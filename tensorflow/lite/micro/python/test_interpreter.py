import os 
import sys 
import numpy as np

curdir = os.path.dirname(os.path.abspath(__file__))

build_dir = os.path.normpath(f'{curdir}/../../../../build/tensorflow/lite/micro/')

sys.path.append(curdir)
sys.path.append(build_dir)

try:
    from .interpreter import Interpreter
except:
    from interpreter import Interpreter

model_path = f'{curdir}/test_data/vww_96_int8.tflite'

interp = Interpreter(model_path)
interp.allocate_tensors()

input_details = interp.get_input_details()[0]
output_details = interp.get_output_details()[0]
input_index = input_details['index']
output_index = output_details['index']

x = np.random.rand(*input_details['shape'])*255 
x = x.astype(np.int8)

input_tensor = interp.tensor(input_index)()[0]
np.copyto(input_tensor, x)
input_tensor = None

interp.invoke()

output = np.squeeze(interp.get_tensor(output_index))

print(f'{output}')