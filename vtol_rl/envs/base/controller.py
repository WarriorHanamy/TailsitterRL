from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class ControllerBase(ABC):
    def __init__(self, control_type: str):
        self.control_type = control_type

    @abstractmethod
    def control(self, goal: torch.Tensor, state: torch.Tensor | None) -> torch.Tensor:
        raise NotImplementedError


class ThrustController(ControllerBase):
    def __init__(self, control_type: str = "thrust"):
        super().__init__(control_type)

    def control(
        self, goal: torch.Tensor, state: torch.Tensor | None = None
    ) -> torch.Tensor:
        return goal


class BodyrateController(ControllerBase):
    def __init__(self, control_type: str = "bodyrate"):
        super().__init__(control_type)

    def control(self, goal: torch.Tensor, state: torch.Tensor | None) -> torch.Tensor:
        self.goal = goal
        return self.goal


class VelocityController(ControllerBase):
    def __init__(self, control_type: str = "velocity"):
        super().__init__(control_type)
        raise NotImplementedError

    def control(self, goal: torch.Tensor, state: torch.Tensor | None) -> torch.Tensor:
        raise NotImplementedError
