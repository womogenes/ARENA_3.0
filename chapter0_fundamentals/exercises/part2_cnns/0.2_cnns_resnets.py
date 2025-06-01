# %%
import json
import sys
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
from IPython.display import display
from jaxtyping import Float, Int
from PIL import Image
from rich import print as rprint
from rich.table import Table
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from tqdm.notebook import tqdm

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part2_cnns"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

MAIN = __name__ == "__main__"

import part2_cnns.tests as tests
import part2_cnns.utils as utils
from plotly_utils import line

# %%

class ReLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.where(x >= 0, x, 0)

tests.test_relu(ReLU)

# %%

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        """
        A simple linear (technically, affine) transformation.

        The fields should be named `weight` and `bias` for compatibility with PyTorch.
        If `bias` is False, set `self.bias` to None.
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        
        init_lo = -1/(in_features**0.5)
        init_hi = 1/(in_features**0.5)

        w = torch.FloatTensor(out_features, in_features).uniform_(init_lo, init_hi)
        self.weight = nn.Parameter(w)

        if bias:
            b = torch.FloatTensor(out_features).uniform_(init_lo, init_hi)
            self.bias = nn.Parameter(b)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        """
        x: shape (*, in_features)
        Return: shape (*, out_features)
        """
        # Reshape x to make it a batch of column vectors
        mul_result = torch.matmul(self.weight, x[..., :, None]).squeeze(dim=2)
        if self.bias is not None:
            mul_result += self.bias
        return mul_result

    def extra_repr(self) -> str:
        return f"in_features={self.in_features},out_features={self.out_features},bias={self.bias is not None}"


tests.test_linear_parameters(Linear, bias=False)
tests.test_linear_parameters(Linear, bias=True)
tests.test_linear_forward(Linear, bias=False)
tests.test_linear_forward(Linear, bias=True)


# %%
class Flatten(nn.Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: Tensor) -> Tensor:
        """
        Flatten out dimensions from start_dim to end_dim, inclusive of both.
        """
        shape = input.shape

        # Get start & end dims, handling negative indexing for end dim
        start_dim = self.start_dim
        end_dim = self.end_dim if self.end_dim >= 0 else len(shape) + self.end_dim

        # Get the shapes to the left / right of flattened dims, as well as the size of the flattened middle
        shape_left = shape[:start_dim]
        shape_right = shape[end_dim + 1 :]
        shape_middle = torch.prod(torch.tensor(shape[start_dim : end_dim + 1])).item()

        return torch.reshape(input, shape_left + (shape_middle,) + shape_right)

    def extra_repr(self) -> str:
        return ", ".join([f"{key}={getattr(self, key)}" for key in ["start_dim", "end_dim"]])


# %%

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.flatten = Flatten()
        self.linear1 = Linear(28*28, 100)
        self.relu = ReLU()
        self.linear2 = Linear(100, 10)

    def forward(self, x: Tensor) -> Tensor:
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


tests.test_mlp_module(SimpleMLP)
tests.test_mlp_forward(SimpleMLP)
