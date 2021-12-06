"""
This script compares tflite_runtime vs tflite_micro_runtime on the RaspberryPi Zero.

Before running this script, run the following PIP commands on your RPI0:

# Install tflite_runtime-2.5.0 (or similar)
pip3 install https://github.com/driedler/tflite_runtime_rpi0w/releases/download/2.5.0/tflite_runtime-2.5.0-cp37-cp37m-linux_armv6l.whl

# Install tflite_micro_runtime-1.2.0
pip3 install https://github.com/driedler/tflite_micro_runtime/releases/download/1.2.0/tflite_micro_runtime-1.2.0-cp37-cp37m-linux_armv6l.whl


Then run the script with your model file:

python3 ./runtime_comparison.py your_model.tflite

This will print the time each interpreter takes to run an inference.
Run the above commands, there was about an 8x improvement with tflite_micro_runtime


"""

import time 
import sys

from tflite_runtime.interpreter import Interpreter as tflite_Interpreter
from tflite_micro_runtime.interpreter import Interpreter as tflm_Interpreter



def main():
    if len(sys.argv) < 2:
        print('Must provide .tflite as argument')
        sys.exit(-1)

    tflite_path = sys.argv[1]
    print(f'Profiling {tflite_path}')

    tflite = tflite_Interpreter(tflite_path)
    tflite.allocate_tensors()
    tflm = tflm_Interpreter(tflite_path)
    tflm.allocate_tensors()

    tflite_time = 0
    tflm_time = 0

    for i in range(10):
        start = time.time()
        tflite.invoke()
        elapsed = time.time() - start 
        tflite_time += elapsed
        print(f'TFLITE-{i}: {elapsed}s')

        start = time.time()
        tflm.invoke()
        elapsed = time.time() - start 
        tflm_time += elapsed
        print(f'  TFLM-{i}: {elapsed}s')
        
    tflite_avg = tflite_time / 10
    tflm_avg = tflm_time / 10
    ratio = tflite_avg / tflm_avg
        
    print(f'tflite_runtime avg: {tflite_avg}s')
    print(f'tflite_micro_runtime avg: {tflm_avg}s')
    print(f'tflite_micro_runtime is {ratio:.1f}x faster')

if __name__ == '__main__':
    main()
