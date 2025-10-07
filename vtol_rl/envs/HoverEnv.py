from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from .base.tailsitterGymEnvs import TailsitterEnvsBase
from vtol_rl.utils.logger import setup_logger
from vtol_rl.utils.randomization import UniformStateRandomizer
from vtol_rl.utils.type import TensorDict

UniformStateRandomizer


class HoverEnv(TailsitterEnvsBase):
    logger = setup_logger("HoverEnv", log_file="HoverEnv.log")

    def __init__(
        self,
        num_agent_per_scene: int = 1,
        num_scene: int = 1,
        seed: int = 42,
        requires_grad: bool = False,
        random_kwargs: Mapping[str, Any] | None = None,
        dynamics_kwargs: Mapping[str, Any] | None = None,
        scene_kwargs: Mapping[str, Any] | None = None,
        sensor_kwargs: Sequence[Mapping[str, Any]] | None = None,
        device: str = "cpu",
        visual: bool = False,
        target: torch.Tensor | None = None,
        max_episode_steps: int = 256,
        tensor_output: bool = False,
    ):
        random_kwargs = (
            {
                "state_generator": {
                    "class": UniformStateRandomizer,
                    "kwargs": {
                        "position": {
                            "mean": [1.0, 0.0, 1.5],
                            "radius": [1.0, 1.0, 0.5],
                        },
                        "orientation": {
                            "mean": [1.0, 0.0, 0.0, 0.0],
                            "radius": [0.0, 0.0, 0.0, 0.0],
                        },
                        "velocity": {
                            "mean": [0.0, 0.0, 0.0],
                            "radius": [0.0, 0.0, 0.0],
                        },
                        "angular_velocity": {
                            "mean": [0.0, 0.0, 0.0],
                            "radius": [0.0, 0.0, 0.0],
                        },
                    },
                }
            }
            if random_kwargs is None
            else random_kwargs
        )

        super().__init__(
            num_agent_per_scene=num_agent_per_scene,
            num_scene=num_scene,
            seed=seed,
            requires_grad=requires_grad,
            random_kwargs=random_kwargs,
            dynamics_kwargs={} if dynamics_kwargs is None else dict(dynamics_kwargs),
            sensor_kwargs=list(sensor_kwargs) if sensor_kwargs is not None else [],
            scene_kwargs={} if scene_kwargs is None else dict(scene_kwargs),
            device=device,
            max_episode_steps=max_episode_steps,
            tensor_output=tensor_output,
        )

        self.target = torch.ones((self.num_envs, 1)) @ torch.as_tensor(
            [1, 0.0, 1.5] if target is None else target
        ).reshape(1, -1)
        self.success_radius = 0.5

    def get_observation(self, indices: torch.Tensor | slice | int | list[int] | None = None) -> TensorDict:
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
            + (self.position - self.target).norm(dim=1)
            * pos_factor  # minimize position error
            + (self.orientation - torch.tensor([1, 0, 0, 0])).norm(dim=1)
            * -0.00001  # minimize orientation error
            + (self.velocity - 0).norm(dim=1) * -0.002  # minimize velocity
            + (self.angular_velocity - 0).norm(dim=1)
            * -0.002  # minimize angular velocity
        )
        return reward
