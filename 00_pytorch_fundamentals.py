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

print(ten_zeroes)
