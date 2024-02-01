import torch

print(torch.version.__version__)
print(torch.cuda.get_device_name(0))
print(torch.cuda.is_available())