import torch

x = torch.tensor([1, 2, 3, 4, 5]).long()
y = torch.tensor([1, 2, 5, 5, 5]).long()

print(x == y)

z = torch.sum(x == y)
print(z)
