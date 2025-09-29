import numpy as np
from .base.droneGymEnv import DroneGymEnvsBase
from typing import Optional, Dict
import torch as th
from gymnasium import spaces

from vtol_rl.utils.type import TensorDict


class LandingEnv(DroneGymEnvsBase):
    def __init__(
        self,
        num_agent_per_scene: int = 1,
        num_scene: int = 1,
        seed: int = 42,
        visual: bool = False,
        requires_grad: bool = False,
        random_kwargs: dict = {},
        dynamics_kwargs: dict = {},
        scene_kwargs: dict = {},
        sensor_kwargs: list = [],
        device: str = "cpu",
        target: Optional[th.Tensor] = None,
        max_episode_steps: int = 128,
        is_eval: bool = False,
    ):
        random_kwargs = {
            "state_generator": {
                "class": "Uniform",
                "kwargs": [
                    {"position": {"mean": [2.0, 0.0, 2.5], "half": [1.0, 1.0, 1.0]}},
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
            dynamics_kwargs=dynamics_kwargs,
            scene_kwargs=scene_kwargs,
            sensor_kwargs=sensor_kwargs,
            device=device,
            max_episode_steps=max_episode_steps,
        )

        self.target = th.tensor([2, 0, 0], device=device)
        self.success_radius = 0.5
        self.observation_space["target"] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
        )
        self.centers = None
        self.max_sense_radius = 10.0

    def get_failure(self):
        target_xy = self.target[:2].unsqueeze(0).to(self.device)
        offset = target_xy - self.position[:, :2]
        return offset.norm(dim=1) > self.max_sense_radius

    def get_observation(self, indices=None) -> Dict:
        target_xy = self.target[:2].unsqueeze(0).to(self.device)
        offset = target_xy - self.position[:, :2]
        self.centers = (offset / self.max_sense_radius).clamp(-1.0, 1.0)

        return TensorDict(
            {
                "state": self.state,
                "target": self.centers,
            }
        )

    def get_success(self) -> th.Tensor:
        landing_half = 0.3
        # return th.full((self.num_envs,), False)
        return (
            (self.position[:, 2] <= 0.2)
            & (self.position[:, :2] < (self.target[:2] + landing_half)).all(dim=1)
            & (self.position[:, :2] > (self.target[:2] - landing_half)).all(dim=1)
            & (self.velocity.norm(dim=1) <= 0.3)
        )  #
        # & \
        # ((self.position[:, :2] < self.target[:2] + landing_half).all(dim=1) & (self.position[:, :2] > self.target[:2] - landing_half).all(dim=1))

    def get_reward(self) -> th.Tensor:
        if self.centers is None:
            return th.zeros((self.num_envs,), device=self.device)
        """reward function"""
        reward = (
            0.2 * (1.25 - self.centers.norm(dim=1) / 1).clamp_max(1.0)
            + (self.orientation[:, [0, 1]]).norm(dim=1) * -0.2
            + 0.1 * (3 - self.position[:, 2]).clamp(0, 3) / 3 * 2
            + -0.02 * self.velocity.norm(dim=1)
            + -0.01 * self.angular_velocity.norm(dim=1)
            + 0.1
            * 20
            * self._success
            * (
                10
                + (
                    th.tensor(self.max_episode_steps, device=self.device)
                    - th.tensor(self._step_count, device=self.device)
                )
            )
            / (1 + 2 * self.velocity.norm(dim=1))
        )  # / (self.velocity.norm(dim=1) + 1)

        return reward


class LandingEnv2(LandingEnv):
    def __init__(
        self,
        num_agent_per_scene: int = 1,
        num_scene: int = 1,
        seed: int = 42,
        visual: bool = False,
        requires_grad: bool = False,
        random_kwargs: dict = {},
        dynamics_kwargs: dict = {},
        scene_kwargs: dict = {},
        sensor_kwargs: list = [],
        device: str = "cpu",
        target: Optional[th.Tensor] = None,
        max_episode_steps: int = 128,
        is_eval: bool = False,
    ):
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
            target=target,
            max_episode_steps=max_episode_steps,
            is_eval=is_eval,
        )
        self.target = th.ones((self.num_envs, 1)) @ th.as_tensor(
            [2.0, 0.0, 2.5] if target is None else target
        ).reshape(1, -1)
        if is_eval:
            self.target = th.as_tensor(
                [[2.0, 1.0, 2.5], [2.0, 0.0, 2.5], [2.0, -1.0, 2.5]]
            )
        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32
                ),
            }
        )

    def get_failure(self) -> th.Tensor:
        return self.is_collision

    def get_reward(self) -> th.Tensor:
        eta = th.as_tensor(1.2)
        v_l = 1 * (self.position[:, 2] - 0).clip(min=0.05, max=1).clone().detach()
        descent_v = -self.velocity[:, 2] - 0
        # r_z_punish = ((descent_v > v_l) | (descent_v < 0))
        # r_z = r_z_punish * r_p + \
        #       ~r_z_punish * (eta.pow(descent_v / v_l) - 1) / (eta-1) * 0.1

        r_z_first = descent_v <= v_l
        r_z = (
            ~r_z_first * (eta.pow(-4 * descent_v / v_l + 5) - 1) / (eta - 1) * 0.1
            + r_z_first * (eta.pow(descent_v / v_l) - 1) / (eta - 1) * 0.1
        )

        rho = th.as_tensor(1.2)
        d_s = 2.0 * (self.position[:, 2] - 0).clip(min=0.05, max=1).clone().detach()
        d_xy = (self.target - self.position)[:, :2].norm(dim=1) - 0
        r_xy = (rho.pow(1 - d_xy / d_s) - 1) / (rho - 1) * 0.1

        # toward_v = ((self.velocity[:, :2] -0)* ((self.target - self.position)[:, :2])).sum(dim=1) / d_xy
        # r_xy_is_first_sec = toward_v <= v_l
        # r_xy = r_xy_is_first_sec * 0.1 * (rho.pow(toward_v/v_l)-1)/(rho-1)+ \
        #     ~r_xy_is_first_sec * 0.1*(rho.pow(-4 * descent_v / v_l + 5) - 1) / (rho-1) * 0.1

        r_s = 20.0
        r_l = self.success * r_s + self.failure * -0.1
        reward = 1.0 * r_l + 1.0 * r_xy + 1.0 * r_z

        return reward

    def get_observation(self, indices=None) -> Dict:
        state = th.hstack(
            [
                (self.target - self.position) / self.max_sense_radius,
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
