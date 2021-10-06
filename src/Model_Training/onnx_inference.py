#!/usr/bin/env python3
import numpy as np
import sys
import onnx
import onnxruntime
from onnx import numpy_helper

model = 'keras_model.onnx'

try:
    input_row = sys.argv[1]
    print(input_row)
except ValueError as e:
    print(e)

a = input_row.split(',')
input_array = np.array(a).astype('float32')
session = onnxruntime.InferenceSession(model, None)
input_name = session.get_inputs()[0]
output_name = session.get_outputs()[0]
print(input_name)
print(output_name)
print(input_name.shape)
# input_array = input_array.reshape((1,input_name.shape[1]))
print(input_array.shape)
result = session.run([output_name], {input_name: input_array})
prediction = int(np.argmax(np.array(result).squeeze(), axis=0))
print(prediction)
