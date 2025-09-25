from dataclasses import dataclass
import torch
from typing import Dict, Union, Any
from enum import Enum
import numpy as np


@dataclass
class bound:
    min: float
    max: float


class ACTION_TYPE(Enum):
    THRUST = 0
    BODYRATE = 1
    VELOCITY = 2
    POSITION = 3


action_type_alias: Dict = {
    "thrust": ACTION_TYPE.THRUST,
    "bodyrate": ACTION_TYPE.BODYRATE,
    "velocity": ACTION_TYPE.VELOCITY,
    "position": ACTION_TYPE.POSITION,
}


class Uniform:
    mean: Union[float, torch.Tensor] = 0
    radius: Union[float, torch.Tensor] = 1

    def __init__(
        self,
        mean,
        radius,
    ):
        self.mean = torch.atleast_1d(torch.as_tensor(mean))
        self.radius = torch.atleast_1d(torch.as_tensor(radius))

    def to(self, device):
        self.mean = self.mean.to(device)
        self.radius = self.radius.to(device)
        return self

    def sample(self, size):
        return (torch.rand(size, len(self.mean)) - 0.5) * 2 * self.radius + self.mean


class Normal:
    mean: Union[float, torch.Tensor] = 0
    std: Union[float, torch.Tensor] = 0

    def __init__(
        self,
        mean,
        std,
    ):
        self.mean = torch.as_tensor(mean)
        self.std = torch.as_tensor(std)

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def sample(self, size):
        return torch.normal(self.mean, self.std, size)


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
