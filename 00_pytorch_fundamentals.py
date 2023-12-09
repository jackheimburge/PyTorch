import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

scalar = torch.tensor(7)

vector = torch.tensor([7, 7])

MATRIX = torch.tensor([[7, 8],
                       [9, 10],
                       [10, 10]])

TENSOR = torch.tensor([[[1, 2, 3],
                        [4, 5, 6],
                        [2, 4, 5]]])

## Random tensors are import because neural network learn by starting with tensors full of random data and then adjust
## those random numbers to better represent the data

## random numbers ==> look at data ==> update random numbers ==> look at data ==> update

random_tensor = torch.rand(3, 4)
random_image_size_tensor = torch.rand(1, 224, 224)  ## colour channels, height, width

zero = torch.zeros(3, 4)
one = torch.ones(3, 4)
one_to_ten = torch.arange(start=0, end=10, step=1)
ten_zeroes = torch.ones_like(input=one_to_ten)

float_16_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=torch.float16,
                               device=None,
                               requires_grad=False) ## track gradients or not

## 3 big errors with PyTorch & deep learning include:
## 1. tensors are not the correct datatype (tensor.dtype)
## 2. tensors are not the correct shape (tensor.shape)
## 3. tensors not on the right device (tensor.device)

## manipulating tensors (tensor operations)
## - Addition, Subtraction, Multiplication (element-wise), Division, **Matrix multiplication (dot product)

tensor = torch.tensor([1, 2, 3])

## matrix multiplication
    ## - inner dimension must match:
    # (3, 2) * (3, 2) wont work
    # (2, 3) * (3, 2) will work

    ## - the resulting matrix has the shape of the outer dimensions

tensor_matmul = torch.matmul(tensor, tensor)

tensor_A = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 6]])

tensor_B = torch.tensor([[7, 10],
                         [8, 11],
                         [9, 12]])


matmul_AB = torch.mm(tensor_A, tensor_B.T)

# print(f'Original shapes: tensor_A: {tensor_A.shape}, tensor_B: {tensor_B.shape} ')
# print(f'new shapes: tensor_A: {tensor_A.shape}, tensor_B.T: {tensor_B.T.shape} ** inner dimensions match')
# print(f'Multiplying {tensor_A.shape} * {tensor_B.T.shape}')
# print(f'Output: {matmul_AB}')
# print(f'output shape: {matmul_AB.shape}')

## to fix tensor shape errors. we can manipulate the shape of one of our tensors using a transpose
## a transpose switches the axes or dimensions of a tensor

## finding the min, max, mean, sum of tensor (aggregation)

x = torch.arange(0, 100, 10)
x.min()
x.type(torch.float32).mean()
print(x.sum())

