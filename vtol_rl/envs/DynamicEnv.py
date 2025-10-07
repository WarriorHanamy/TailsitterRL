from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from .base.droneGymEnv import DroneGymEnvsBase
from vtol_rl.utils.type import TensorDict


class DynEnv(DroneGymEnvsBase):
    def __init__(
        self,
        num_agent_per_scene: int = 1,
        num_scene: int = 1,
        seed: int = 42,
        visual: bool = True,
        requires_grad: bool = False,
        random_kwargs: Mapping[str, Any] | None = None,
        dynamics_kwargs: Mapping[str, Any] | None = None,
        scene_kwargs: Mapping[str, Any] | None = None,
        sensor_kwargs: Sequence[Mapping[str, Any]] | None = None,
        device: str = "cpu",
        target: torch.Tensor | None = None,
        max_episode_steps: int = 256,
        tensor_output: bool = False,
    ):
        random_kwargs = {} if random_kwargs is None else dict(random_kwargs)
        dynamics_kwargs = {} if dynamics_kwargs is None else dict(dynamics_kwargs)
        scene_kwargs = {} if scene_kwargs is None else dict(scene_kwargs)
        sensor_kwargs = list(sensor_kwargs) if sensor_kwargs is not None else []

        super().__init__(
            num_agent_per_scene=num_agent_per_scene,
            num_scene=num_scene,
            seed=seed,
            visual=visual,
            requires_grad=requires_grad,
            random_kwargs=random_kwargs,
            dynamics_kwargs=dynamics_kwargs,
            scene_kwargs=scene_kwargs,
            sensor_kwargs=sensor_kwargs,
            device=device,
            max_episode_steps=max_episode_steps,
            tensor_output=tensor_output,
        )

    def get_observation(
        self, indices: torch.Tensor | slice | int | list[int] | None = None
    ) -> TensorDict:
        obs = TensorDict(
            {
                "state": self.state,
            }
        )

        return obs

    def get_success(self) -> torch.Tensor:
        return torch.full((self.num_agent,), False)

    def get_reward(self) -> torch.Tensor:
        base_r = 0.1
        pos_factor = -0.1 * 1 / 9
        reward = (
            base_r
            + (self.position - 0).norm(dim=1) * pos_factor
            + (self.orientation - torch.tensor([1, 0, 0, 0])).norm(dim=1) * -0.00001
            + (self.velocity - 0).norm(dim=1) * -0.002
            + (self.angular_velocity - 0).norm(dim=1) * -0.002
        )

        return reward
