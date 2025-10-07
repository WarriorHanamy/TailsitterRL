import torch

tensor1 = torch.randn(10, 4)

tensor_sub1 = tensor1[:, 0]
tensor_sub2 = tensor1[:, 1:]

tensor_new = torch.cat((tensor_sub1.unsqueeze(1), tensor_sub2), dim=1)
torch_nn = torch.cat((tensor1[:, 0:1], tensor1[:, 1:]), dim=1)
torch_nn1 = torch.cat((tensor_sub1, tensor1[:, 1:]), dim=1)


print(
    f"tensor1: {tensor1}\n, tensor_sub1\n: {tensor_sub1}, tensor_sub1.shape\n: {tensor_sub1.shape}"
)
