from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import torch


@dataclass
class bound:
    min: float
    max: float


class ACTION_TYPE(Enum):
    THRUST = 0
    BODYRATE = 1
    VELOCITY = 2
    POSITION = 3


action_type_alias: dict = {
    "thrust": ACTION_TYPE.THRUST,
    "bodyrate": ACTION_TYPE.BODYRATE,
    "velocity": ACTION_TYPE.VELOCITY,
    "position": ACTION_TYPE.POSITION,
}


class Uniform:
    def __init__(
        self,
        mean: float | torch.Tensor | list | np.ndarray,
        radius: float | torch.Tensor | list | np.ndarray,
    ):
        """
        Args:
            mean: center of the uniform distribution
                Type: torch.Tensor finally, shape (n,).  If input is float, list, or np.ndarray, it will be converted to torch.Tensor.
            radius: half the width of the uniform distribution
                Type: torch.Tensor finally, shape (n,).  If input is float, list, or np.ndarray, it will be converted to torch.Tensor.
        """
        self.mean = torch.as_tensor(mean).view(1, -1)
        self.radius = torch.as_tensor(radius).view(1, -1)

    @classmethod
    def from_min_max(cls, min_val, max_val):
        min_val = torch.as_tensor(min_val)
        max_val = torch.as_tensor(max_val)
        mean = (min_val + max_val) / 2
        radius = (max_val - min_val) / 2
        return cls(mean, radius)

    def to(self, device):
        self.mean = self.mean.to(device)
        self.radius = self.radius.to(device)
        return self

    @property
    def min(self):
        """
        return the minimum value of the uniform distribution
        Args:
            self.mean: shape (N,1)
            self.radius: shape (N,1)
        Returns:
            min: shape (N, 1)
        """
        return self.mean - self.radius

    @property
    def max(self):
        """
        return the maximum value of the uniform distribution
        Args:
            self.mean: shape (N,1)
            self.radius: shape (N,1)
        Returns:
            max: shape (N, 1)
        """
        return self.mean + self.radius

    def sample(self, size):
        return (torch.rand(size, len(self.mean)) - 0.5) * 2 * self.radius + self.mean

    def per_sample_normalize(self, x_orig: torch.Tensor):
        """
        Normalize the input [min, max] to [-1, 1]

        Args:
            x_orig (torch.Tensor): input tensor in original range [min, max]
                Shape: (batch_size, feats_size)
        Returns:
            normed_x (torch.Tensor): normalized tensor in range [-1, 1]
        """
        print(f"x_orig.shape: {x_orig.shape}")

        # This Operation is to normalized to [-1, 1]
        return 2 * (x_orig - self.min) / (self.max - self.min) - 1

        # This Operation is to normalized to [0, 1]
        return (x_orig - self.min) / (self.max - self.min)

    def per_sample_denormalize(self, x_normed: torch.Tensor):
        """
        Denormalize the input [-1, 1] to [min, max]

        Args:
            x_normed (torch.Tensor): input tensor in normalized range [-1, 1]
                Shape: (batch_size, feats_size)
        Returns:
            x_orig (torch.Tensor): denormalized tensor in original range [min, max]
        """
        # This Operation is to denormalized from [-1, 1] to [min, max]
        return x_normed * self.radius + self.mean
        # Another Implementation:
        # return (x_normed + 1) / 2 * (self.max - self.min) + self.min


class Gaussian:
    def __init__(
        self,
        mean: float | torch.Tensor | list | np.ndarray,
        std: float | torch.Tensor | list | np.ndarray,
    ):
        """
        Args:
            mean: center of the gaussian distribution
                Type: torch.Tensor finally, shape (n,). If input is float, list, or np.ndarray, it will be converted to torch.Tensor.
            std: standard deviation of the gaussian distribution
                Type: torch.Tensor finally, shape (n,). If input is float, list, or np.ndarray, it will be converted to torch.Tensor.
        """
        self.mean = torch.as_tensor(mean).view(1, -1)
        self.std = torch.as_tensor(std).view(1, -1)

    @classmethod
    def from_min_max(cls, min_val, max_val):
        """
        Create Normal distribution from min/max values by assuming:
        - mean is midpoint between min and max
        - 99.7% of values fall within min/max (3 standard deviations each side)
        """
        min_val = torch.as_tensor(min_val)
        max_val = torch.as_tensor(max_val)
        mean = (min_val + max_val) / 2
        std = (max_val - min_val) / 6  # 3 std deviations each side
        return cls(mean, std)

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def sample(self, size):
        """Sample from normal distribution"""
        return torch.randn(size, len(self.mean)) * self.std + self.mean

    def per_sample_normalize(self, x_orig: torch.Tensor):
        """
        Normalize the input to approximately standard gaussian (mean=0, std=1)

        Args:
            x_orig (torch.Tensor): input tensor in original range
                Shape: (batch_size, feats_size)
        Returns:
            normed_x (torch.Tensor): normalized tensor with ~Gaussian(0, 1)
        """
        return (x_orig - self.mean) / self.std

    def per_sample_denormalize(self, x_normed: torch.Tensor):
        """
        Denormalize the input from standard normal back to original distribution

        Args:
            x_normed (torch.Tensor): input tensor in normalized range ~Gaussian(0, 1)
                Shape: (batch_size, feats_size)
        Returns:
            x_orig (torch.Tensor): denormalized tensor in original range
        """
        return x_normed * self.std + self.mean


@dataclass
class PID:
    p: torch.Tensor = torch.diag(torch.tensor([1, 1, 1]))
    i: torch.Tensor = torch.diag(torch.tensor([1, 1, 1]))
    d: torch.Tensor = torch.diag(torch.tensor([1, 1, 1]))

    def to(self, device):
        self.p = self.p.to(device)
        self.i = self.i.to(device)
        self.d = self.d.to(device)
        return self

    def clone(self):
        self.p = self.p.clone()
        self.i = self.i.clone()
        self.d = self.d.clone()

        return self

    def detach(self):
        self.p = self.p.detach()
        self.i = self.i.detach()
        self.d = self.d.detach()

        return self


class SortDict(dict):
    def __init__(self, data):
        super().__init__(data)

    def __getitem__(self, __key: Any) -> Any:
        if isinstance(__key, str):
            return super().__getitem__(__key)
        else:
            # return super().__getitem__(list(self.keys())[__key])
            return dict([(key, super().__getitem__(key)[__key]) for key in self.keys()])


# create a dict designed for tensor that can use detach
class TensorDict(dict):
    def __init__(self, data):
        super().__init__(data)

    # return a new detach, do not change instance itself
    def detach(self):
        return TensorDict({key: self[key].detach() for key in self.keys()})

    def clone(self):
        for key in self.keys():
            self[key] = self[key].clone()

        return self

    def __getitem__(self, key: Any, keepdim=False) -> Any:
        if isinstance(key, str):
            return super().__getitem__(key)
        elif isinstance(key, int):
            return TensorDict({k: torch.atleast_2d(v[key]) for k, v in self.items()})
        elif hasattr(key, "__iter__"):
            # Convert key to CPU if it's a tensor to avoid device mismatch
            if hasattr(key, "cpu"):
                key = key.cpu()
            return TensorDict({k: torch.atleast_2d(v[key]) for k, v in self.items()})
        else:
            raise TypeError("Invalid key type. Must be either str or int.")

    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(key, str):
            super().__setitem__(key, value)
        elif isinstance(key, (int, torch.Tensor, np.ndarray, list)):
            for k in self.keys():
                self[k][key] = value[k]
        else:
            raise TypeError("Invalid key type. Must be either str or int.")

    def append(self, data):
        if isinstance(data, TensorDict):
            for key, value in data.items():
                self[key] = torch.cat([self[key], data[key]])

    def cpu(self):
        for key, value in self.items():
            self[key] = self[key].cpu()
        return self

    def as_tensor(self, device=torch.device("cpu")):
        d = TensorDict({})
        for key, value in self.items():
            d[key] = torch.as_tensor(value, device=device)

        return d

    def to(self, device):
        for key, value in self.items():
            self[key] = value.to(device)
        return self

    def reshape(self, shape):
        for key, value in self.items():
            self[key] = value.reshape(shape)
        return self

    @staticmethod
    def stack(x_list):
        keys = x_list[0].keys()
        r = TensorDict({})

        for key in keys:
            cache = []
            for x in x_list:
                cache.append(x[key])
            r[key] = torch.stack(cache)
            # r[key] = torch.reshape(r[key], (-1, *r[key].shape[2:]))
        return r

    def numpy(self):
        for key, value in self.items():
            self[key] = self[key].cpu().detach().numpy()
        return self

    def __len__(self):
        lens = [len(value) for value in self.values()]
        # assert all lens equal
        assert all([len_ == lens[0] for len_ in lens])
        return lens[0]

    def __iter__(self):
        # 获取第一个 value 的长度
        first_length = len(self)
        # 生成每个索引对应的字典
        for i in range(first_length):
            yield self[i]
