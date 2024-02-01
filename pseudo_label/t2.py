import torch

x = torch.tensor([0, 10, 1]).float()
y = torch.tensor([1, 2, 3])
# print(x)
#
# z = torch.argmax(x, dim = 1)
# print(z)
# x =

z = [x, y]
print(torch.cat(z,dim = 0))

