import torch
from torch import nn  ## building blocks for neural networks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## 1 Data (preparing and loading)

## linear regressions formula (Y = a + bX) where X is the explanatory variable and Y is the dependent variable.
# The slope of the line is b, and a is the intercept

weight = 0.7  # b
bias = 0.3  # a

start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = bias + (weight * X)  # Y = a + bX

print(X[:10], y[:10])

# 3 datasets in ML

# 1. training set (60-80%) always
# 2. validation set (10-20%) often
# 3. testing set (10-20%) always

# splitting data into sets

train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]


def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=None):
    """
    plots training data, test data and compares predictions
    """
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training Data")
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    plt.legend(prop={"size": 14})
    plt.show()


# generalization: the ability for a machine learning model to perform well on data it hasn't seen before

plot_predictions()


# Building models

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1,
                                                requires_grad=True,
                                                dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1,
                                             requires_grad=True,
                                             dtype=torch.float))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.weights * x + self.bias
