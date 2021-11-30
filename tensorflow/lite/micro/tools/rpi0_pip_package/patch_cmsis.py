import os 
import sys 

curdir = os.path.dirname(os.path.abspath(__file__))
cmsis_arm_math_types_h_path = f'{curdir}/../make/downloads/cmsis/CMSIS/DSP/Include/arm_math_types.h'


with open(cmsis_arm_math_types_h_path, 'r') as fp:
    data = fp.read()

if '#define memset __builtin_memset' in data:
    sys.exit()


# Ensure the builtin memset and memcpy
# are used. This greatly reduce the overhead of the kernels.
patch_data = '#define _ARM_MATH_TYPES_H_\n\n'
patch_data += '#define memset __builtin_memset\n'
patch_data += '#define memcpy __builtin_memcpy\n\n'

data = data.replace('#define _ARM_MATH_TYPES_H_', patch_data)

with open(cmsis_arm_math_types_h_path, 'w') as fp:
    fp.write(data)