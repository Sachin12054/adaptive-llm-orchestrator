import torch

print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Count:", torch.cuda.device_count())
    print("Device Name:", torch.cuda.get_device_name(0))
