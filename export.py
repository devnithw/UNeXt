import torch
import torch.onnx
from model import UNext
import argparse
import os

# Argument parsing
parser = argparse.ArgumentParser(description='Exporting the model to ONNX format')
parser.add_argument('--input', type=str, required=True, help='Input model weights')
parser.add_argument('--output', type=str, required=True, help='Output ONNX file name')
args = parser.parse_args()

# Initialize the model
model = UNext(num_classes=1)

# Import the model weights
MODEL_LOAD_PATH = f'models/saved_models/{args.input}.pth'
# Output folder
MODEL_OUTPUT_PATH = f'exported_models/{args.output}.onnx'
os.makedirs('exported_models', exist_ok=True)

# Load the pretrained weights
model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=torch.device('cpu')))

# Input to the model
model.eval()
# Dummy input
x = torch.randn(1, 3, 256, 256, requires_grad=True)
torch_out = model(x)

# Export the model
torch.onnx.export(model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  MODEL_OUTPUT_PATH,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  input_names = ['input'],   # the model's input names
                  output_names = ['output']) # the model's output names