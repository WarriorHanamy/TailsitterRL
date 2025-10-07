from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch as th
from gymnasium import spaces
import numpy as np

from .base.droneGymEnv import DroneGymEnvsBase
from vtol_rl.utils.type import TensorDict


class TrackEnv(DroneGymEnvsBase):
    def __init__(
        self,
        num_agent_per_scene: int = 1,
        num_scene: int = 1,
        seed: int = 42,
        visual: bool = False,
        requires_grad: bool = False,
        random_kwargs: Mapping[str, Any] | None = None,
        dynamics_kwargs: Mapping[str, Any] | None = None,
        scene_kwargs: Mapping[str, Any] | None = None,
        sensor_kwargs: Sequence[Mapping[str, Any]] | None = None,
        device: str = "cpu",
        target: th.Tensor | None = None,
        max_episode_steps: int = 256,
        latent_dim=None,
        tensor_output=False,
    ):
        self.center = th.as_tensor([2, 0, 1])
        self.next_points_num = 10
        self.radius = 2
        self.dt = 0.1
        self.radius_spd = 0.2 * th.pi / 1
        self.success_radius = 0.5

        random_kwargs = {
            "state_generator": {
                "class": "Uniform",
                "kwargs": [
                    {
                        "position": {
                            "mean": [self.center[0], 0.0, self.center[2]],
                            "half": [0.2, 0.2, 0.2],
                        }
                    },
                ],
            }
        }

        super().__init__(
            num_agent_per_scene=num_agent_per_scene,
            num_scene=num_scene,
            seed=seed,
            visual=visual,
            requires_grad=requires_grad,
            random_kwargs=random_kwargs,
            dynamics_kwargs={} if dynamics_kwargs is None else dict(dynamics_kwargs),
            sensor_kwargs=list(sensor_kwargs) if sensor_kwargs is not None else [],
            scene_kwargs={} if scene_kwargs is None else dict(scene_kwargs),
            device=device,
            max_episode_steps=max_episode_steps,
            tensor_output=tensor_output,
        )

        self.observation_space["state"] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                3 * (self.next_points_num - 1)
                + self.observation_space["state"].shape[0],
            ),
            dtype=np.float32,
        )

        self.update_target()

    def update_target(self):
        ts = (
            self.t.repeat(self.next_points_num, 1).T
            + th.arange(self.next_points_num) * self.dt
        )
        self.target = (
            th.stack(
                [
                    self.radius * th.cos(self.radius_spd * ts) + self.center[0],
                    self.radius * th.sin(self.radius_spd * ts) + self.center[1],
                    0 * th.sin(self.radius_spd * ts) + self.center[2],
                ]
            )
        ).permute(1, 2, 0)

    def get_observation(
        self, indices: th.Tensor | slice | int | list[int] | None = None
    ) -> TensorDict:
        diff_pos = self.target - self.position.unsqueeze(1)
        # consider target as next serveral waypoint
        diff_pos_flatten = diff_pos.reshape(self.num_envs, -1)

        state = th.hstack(
            [
                diff_pos_flatten / self.max_sense_radius,
                self.orientation,
                self.velocity / 10,
                self.angular_velocity / 10,
            ]
        ).to(self.device)

        return TensorDict(
            {
                "state": state,
            }
        )

    def get_success(self) -> th.Tensor:
        return th.full((self.num_agent,), False)
        # return (self.position - self.target).norm(dim=1) < self.success_radius

    def get_reward(self) -> th.Tensor:
        base_r = 0.1
        pos_factor = -0.1 * 1 / 9
        reward = (
            base_r
            + (self.position - self.target[:, 0, :]).norm(dim=1) * pos_factor
            + (self.orientation - th.tensor([1, 0, 0, 0])).norm(dim=1) * -0.00001
            + (self.velocity - 0).norm(dim=1) * -0.002
            + (self.angular_velocity - 0).norm(dim=1) * -0.002
        )

        return reward


class TrackEnv2(TrackEnv):
    def __init__(
        self,
        num_agent_per_scene: int = 1,
        num_scene: int = 1,
        seed: int = 42,
        visual: bool = False,
        requires_grad: bool = False,
        random_kwargs: Mapping[str, Any] | None = None,
        dynamics_kwargs: Mapping[str, Any] | None = None,
        scene_kwargs: Mapping[str, Any] | None = None,
        sensor_kwargs: Sequence[Mapping[str, Any]] | None = None,
        device: str = "cpu",
        target: th.Tensor | None = None,
        max_episode_steps: int = 256,
        latent_dim=None,
        tensor_output=False,
    ):
        super().__init__(
            num_agent_per_scene=num_agent_per_scene,
            num_scene=num_scene,
            seed=seed,
            visual=visual,
            requires_grad=requires_grad,
            random_kwargs=random_kwargs,
            dynamics_kwargs={} if dynamics_kwargs is None else dict(dynamics_kwargs),
            scene_kwargs={} if scene_kwargs is None else dict(scene_kwargs),
            sensor_kwargs=list(sensor_kwargs) if sensor_kwargs is not None else [],
            device=device,
            max_episode_steps=max_episode_steps,
            target=target,
            latent_dim=latent_dim,
            tensor_output=tensor_output,
        )
        self._depth_resolution = (1, 64, 64)
        self.observation_space["depth"] = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=self._depth_resolution,
            dtype=np.float32,
        )

    def get_observation(
        self, indices: th.Tensor | slice | int | list[int] | None = None
    ) -> TensorDict:
        diff_pos = self.target - self.position.unsqueeze(1)
        # consider target as next serveral waypoint
        diff_pos_flatten = diff_pos.reshape(self.num_envs, -1)

        state = th.hstack(
            [
                diff_pos_flatten / self.max_sense_radius,
                self.orientation,
                self.velocity / 10,
                self.angular_velocity / 10,
            ]
        ).to(self.device)

        depth = self._synthesize_depth()

        return TensorDict(
            {
                "state": state,
                "depth": depth,
            }
        )

    def _synthesize_depth(self) -> th.Tensor:
        distance = (self.target[:, 0, :] - self.position).norm(dim=1, keepdim=True)
        depth = distance.view(-1, 1, 1, 1).repeat(1, *self._depth_resolution[1:])
        return depth.to(self.device)
