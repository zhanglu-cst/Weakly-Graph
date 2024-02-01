import torch

x = torch.tensor([1, 2, 3, ])
y = torch.tensor([4, 5, 6, 7, 8])
z = torch.cat((x, y), dim = 0)
print(z)
