import onnx

# Load the ONNX model
model = onnx.load('models-onnx/llama3.1-8b-onnx/1/model.onnx/model.onnx')

# Get input and output tensors
input_all = [node.name for node in model.graph.input]
output_all = [node.name for node in model.graph.output]

# Print input names and shapes
for input in model.graph.input:
    print(f"Input Name: {input.name}")
    dims = [dim.dim_value for dim in input.type.tensor_type.shape.dim]
    print(f"Shape: {dims}")

# Print output names and shapes
for output in model.graph.output:
    print(f"Output Name: {output.name}")
    dims = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
    print(f"Shape: {dims}")
