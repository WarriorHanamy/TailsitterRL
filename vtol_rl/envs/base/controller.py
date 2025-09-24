import torch
from abc import ABC, abstractmethod
from typing import Optional


class ControllerBase(ABC):
    def __init__(self, control_type: str):
        self.control_type = control_type

    @abstractmethod
    def control(
        self, goal: torch.Tensor, state: Optional[torch.Tensor]
    ) -> torch.Tensor:  # thrusts
        raise NotImplementedError


class ThrustController(ControllerBase):
    def __init__(self, control_type: str = "thrust"):
        super().__init__(control_type)

    def control(
        self, goal: torch.Tensor, state: Optional[torch.Tensor] = None
    ) -> torch.Tensor:  # thrust
        return goal


class BodyrateController(ControllerBase):
    def __init__(self, control_type: str = "bodyrate"):
        super().__init__(control_type)

    def control(
        self, goal: torch.Tensor, state: Optional[torch.Tensor]
    ) -> torch.Tensor:  # thrust
        self.goal = goal
        return self.goal


class VelocityController(ControllerBase):
    def __init__(self, control_type: str = "velocity"):
        super().__init__(control_type)
        raise NotImplementedError

    def control(
        self, goal: torch.Tensor, state: Optional[torch.Tensor]
    ) -> torch.Tensor:  # thrust
        raise NotImplementedError
