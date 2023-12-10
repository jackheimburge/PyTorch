import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(device)
# scalar = torch.tensor(7)
#
# vector = torch.tensor([7, 7])
#
# MATRIX = torch.tensor([[7, 8],
#                        [9, 10],
#                        [10, 10]])
#
# TENSOR = torch.tensor([[[1, 2, 3],
#                         [4, 5, 6],
#                         [2, 4, 5]]])
#
# ## Random tensors are import because neural network learn by starting with tensors full of random data and then adjust
# ## those random numbers to better represent the data
#
# ## random numbers ==> look at data ==> update random numbers ==> look at data ==> update
#
# random_tensor = torch.rand(3, 4)
# random_image_size_tensor = torch.rand(1, 224, 224)  ## colour channels, height, width
#
# zero = torch.zeros(3, 4)
# one = torch.ones(3, 4)
# one_to_ten = torch.arange(start=0, end=10, step=1)
# ten_zeroes = torch.ones_like(input=one_to_ten)
#
# float_16_tensor = torch.tensor([3.0, 6.0, 9.0],
#                                dtype=torch.float16,
#                                device=None,
#                                requires_grad=False) ## track gradients or not
#
# ## 3 big errors with PyTorch & deep learning include:
# ## 1. tensors are not the correct datatype (tensor.dtype)
# ## 2. tensors are not the correct shape (tensor.shape)
# ## 3. tensors not on the right device (tensor.device)
#
# ## manipulating tensors (tensor operations)
# ## - Addition, Subtraction, Multiplication (element-wise), Division, **Matrix multiplication (dot product)
#
# tensor = torch.tensor([1, 2, 3])
#
# ## matrix multiplication
#     ## - inner dimension must match:
#     # (3, 2) * (3, 2) wont work
#     # (2, 3) * (3, 2) will work
#
#     ## - the resulting matrix has the shape of the outer dimensions
#
# tensor_matmul = torch.matmul(tensor, tensor)
#
# tensor_A = torch.tensor([[1, 2],
#                          [3, 4],
#                          [5, 6]])
#
# tensor_B = torch.tensor([[7, 10],
#                          [8, 11],
#                          [9, 12]])
#
#
# matmul_AB = torch.mm(tensor_A, tensor_B.T)
#
# # print(f'Original shapes: tensor_A: {tensor_A.shape}, tensor_B: {tensor_B.shape} ')
# # print(f'new shapes: tensor_A: {tensor_A.shape}, tensor_B.T: {tensor_B.T.shape} ** inner dimensions match')
# # print(f'Multiplying {tensor_A.shape} * {tensor_B.T.shape}')
# # print(f'Output: {matmul_AB}')
# # print(f'output shape: {matmul_AB.shape}')
#
# ## to fix tensor shape errors. we can manipulate the shape of one of our tensors using a transpose
# ## a transpose switches the axes or dimensions of a tensor
#
# ## finding the min, max, mean, sum of tensor (aggregation)
#
# x = torch.arange(0, 100, 10)
# x.min()
# x.type(torch.float32).mean()
# x.argmin()
# x.argmax() ##argmin() argmax() returns index of min/max
#
# ## reshape -- reshapes a tensor to a defined shape
# # view -- return a view of a new tensor, but keeps original tensor in memory
# # stack -- combine multiple tensors on top of each other (vstack) or side by side (hstack)
# # squeeze -- removes all '1' dimensions
# # unsqueeze -- adds '1' dimension
# # permute -- return a view of the input with dimensions swapped in a certain way
#
# z = torch.arange(1., 10.)
#
# z_reshaped = z.reshape(1, 9)
#
# y = z_reshaped.view(1, 9)
# y[:, 0] = 5
#
# z_stacked = torch.stack([z, z, z, z], dim=0)
# # print(f'previous tensor: {z_reshaped}')
# # print(f'previous shape: {z_reshaped.shape}')
# z_squeezed = z_reshaped.squeeze()
#
# # print(f'new tensor: {z_reshaped}')
# # print(f'new shape: {z_reshaped.shape}')
# z_unsqueezed = z_reshaped.unsqueeze(dim=2) # dim is at which index the dimension is added to
#
# #
# # print(z_reshaped.shape, z_unsqueezed.shape)
#
# x_original = torch.rand(224, 224, 3)
#
# # x_permuted = x_original.permute(2, 0, 1) # numbers in argument are representing the orignal index values
#
# # print(x_permuted.shape, x_original.shape)
# # print(x_permuted, x_original)
#
# # x[:, :, 0]  can use : to select all values from target index/dimension
#
# # NumPy --> tensor = torch.from_numpy(ndarray)
# # tensor --> NumPy torch.Tensor.numpy()

# array = np.arange(1.0, 8.0)
# tensor = torch.from_numpy(array).type(torch.float32) # will use float64 by default in PyTorch
# array = array + 1 # wont change values in tensor
#
# # print(f'array from numpy: {array} \ntensor created: {tensor}')
# tensor = torch.ones(7)
# numpy_arr = tensor.numpy()
# tensor = tensor + 1 # numpy_arr will not change
# print(numpy_arr, tensor)

## reproducibility

# to reduce randomness in neural networks PyTorch comes with the concept of random seed
# -- flavors the randomness


random_tensor_A = torch.rand(3, 4)
random_tensor_B = torch.rand(3, 4)

# print(random_tensor_B)
# print(random_tensor_A)
# print(random_tensor_A == random_tensor_B)

# RANDOM_SEED = 42
#
# torch.manual_seed(RANDOM_SEED)
# random_tensor_C = torch.rand(3, 4)
#
# torch.manual_seed(RANDOM_SEED)
# random_tensor_D = torch.rand(3, 4)
#
# print(random_tensor_C)
# print(random_tensor_D)
# print(random_tensor_C == random_tensor_D)

tensor = torch.tensor([1, 2, 3])
tensor_on_mps = tensor.to(device)

tensor_on_cpu = tensor_on_mps.cpu().numpy()

print(tensor_on_cpu, tensor_on_mps)







