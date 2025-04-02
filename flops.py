import torch
from fvcore.nn import FlopCountAnalysis
from modelC import *
from modelD import *

device = "cuda" if torch.cuda.is_available() else "cpu"
input = torch.randn(2, 3, 32, 32).to(device)  # Use batch size of 2

# ModelC
model_c = MobileNetLSTMSTAM(num_classes=10).to(device)
model_c.eval()  # Set the model to evaluation mode
flops_c = FlopCountAnalysis(model_c, input).total()
print(f"ModelC FLOPs: {flops_c / 1e6:.2f}M")

# ModelD
model_d = LightCNN_LSTM_STAM(num_classes=10).to(device)
model_d.eval()  # Set the model to evaluation mode
flops_d = FlopCountAnalysis(model_d, input).total()
print(f"ModelD FLOPs: {flops_d / 1e6:.2f}M")
