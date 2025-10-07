from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch as th

from .base.droneGymEnv import DroneGymEnvsBase
from vtol_rl.utils.randomization import UniformStateRandomizer

g = th.tensor([[0, 0, -9.8]])


class ball:
    def __init__(
        self,
        num_agents,
        random_kwargs=None,
        dt=0.2,
    ):
        self.num_agents = num_agents
        random_kwargs = (
            dict(
                position={"mean": [1.0, 0.0, 1.5], "half": [0.0, 2.0, 1.0]},
                orientation={"mean": [0.0, 0.0, 0.0], "half": [0.0, 0.0, 0.0]},
                velocity={"mean": [0.0, 0.0, 0.0], "half": [1.0, 1.0, 0.0]},
                angular_velocity={"mean": [0.0, 0.0, 0.0], "half": [0.0, 0.0, 0.0]},
            )
            if random_kwargs is None
            else random_kwargs
        )
        self.randomizer = UniformStateRandomizer(**random_kwargs)
        self.position = th.empty((self.num_agents, 3))
        self.orientation = th.empty((self.num_agents, 3))
        self.velocity = th.empty((self.num_agents, 3))
        self.angular_velocity = th.empty((self.num_agents, 3))
        self.is_collision = th.zeros(self.num_agents, dtype=th.bool)

        self.dt = dt

    def reset(self):
        self.reset_by_id()

    def reset_by_id(self, indices=None):
        indices = th.arange(self.num_agents) if indices is None else indices
        pos, _, vel, _ = self.randomizer.generate(len(indices))
        self.position[indices] = pos
        self.velocity[indices] = vel
        self.is_collision[indices] = False

    def step(self):
        self.position += self.velocity * self.dt
        self.velocity += g * self.dt
        self.is_collision = self.position[:, 2] < 0.1


class CatchEnv(DroneGymEnvsBase):
    def __init__(
        self,
        num_agent_per_scene: int = 1,
        num_scene: int = 1,
        seed: int = 42,
        visual: bool = False,
        max_episode_steps: int = 1000,
        device: th.device | str = th.device("cpu"),
        dynamics_kwargs: Mapping[str, Any] | None = None,
        random_kwargs: Mapping[str, Any] | None = None,
        requires_grad: bool = False,
        scene_kwargs: Mapping[str, Any] | None = None,
        sensor_kwargs: Sequence[Mapping[str, Any]] | None = None,
        latent_dim=None,
    ):
        dynamics_kwargs = {} if dynamics_kwargs is None else dict(dynamics_kwargs)
        random_kwargs = {} if random_kwargs is None else dict(random_kwargs)
        scene_kwargs = {} if scene_kwargs is None else dict(scene_kwargs)
        sensor_kwargs = list(sensor_kwargs) if sensor_kwargs is not None else []

        super().__init__(
            num_agent_per_scene=num_agent_per_scene,
            num_scene=num_scene,
            seed=seed,
            visual=visual,
            max_episode_steps=max_episode_steps,
            device=device,
            dynamics_kwargs=dynamics_kwargs,
            random_kwargs=random_kwargs,
            requires_grad=requires_grad,
            scene_kwargs=scene_kwargs,
            sensor_kwargs=sensor_kwargs,
            latent_dim=latent_dim,
        )
        scene_kwargs["object_kwargs"] = {
            "object_setting_path": "VisFly/datasets/visfly-beta/configs/free_falling_objects.json",
            "isolated": True,
        }

    def get_observation(self, indice=None) -> dict:
        pass

    def get_success(self) -> th.Tensor:
        pass

    def get_reward(
        self,
    ) -> np.ndarray | th.Tensor:
        pass
