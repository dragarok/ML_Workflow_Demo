#!/usr/bin/env python3
import numpy as np
import sys
import onnx
import onnxruntime
from onnx import numpy_helper

model = 'keras_model.onnx'

try:
    input_row = sys.argv[1]
except ValueError as e:
    print(e)

# NOTE Uncomment this line if you pass input row as "1,2,3" inside quotes.
# a = input_row.replace('"','')

a = input_row.split(',')
session = onnxruntime.InferenceSession(model)
onnx_input = session.get_inputs()[0]
input_name = onnx_input.name
output_name = session.get_outputs()[0].name
input_array = np.array(a).reshape(1, onnx_input.shape[1]).astype(np.float32)
result = session.run([output_name], {input_name: input_array})
print(result)
prediction = int(np.argmax(np.array(result).squeeze(), axis=0))
print(prediction)
