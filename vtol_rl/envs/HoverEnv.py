from .base.droneGymEnv import DroneGymEnvsBase
from typing import Optional, Dict
import torch
from habitat_sim import SensorType
from vtol_rl.utils.type import TensorDict


class HoverEnv(DroneGymEnvsBase):
    def __init__(
        self,
        num_agent_per_scene: int = 1,
        num_scene: int = 1,
        seed: int = 42,
        visual: bool = True,
        requires_grad: bool = False,
        random_kwargs: dict = None,
        dynamics_kwargs: dict = {},
        scene_kwargs: dict = {},
        sensor_kwargs: list = [],
        device: str = "cpu",
        target: Optional[torch.Tensor] = None,
        max_episode_steps: int = 256,
        tensor_output: bool = False,
    ):
        random_kwargs = (
            {
                "state_generator": {
                    "class": "Uniform",
                    "kwargs": [
                        {
                            "position": {
                                "mean": [1.0, 0.0, 1.5],
                                "half": [1.0, 1.0, 0.5],
                            }
                        },
                    ],
                }
            }
            if random_kwargs is None
            else random_kwargs
        )

        super().__init__(
            num_agent_per_scene=num_agent_per_scene,
            num_scene=num_scene,
            seed=seed,
            visual=visual,
            requires_grad=requires_grad,
            random_kwargs=random_kwargs,
            dynamics_kwargs=dynamics_kwargs,
            sensor_kwargs=sensor_kwargs,
            scene_kwargs=scene_kwargs,
            device=device,
            max_episode_steps=max_episode_steps,
            tensor_output=tensor_output,
        )

        self.target = torch.ones((self.num_envs, 1)) @ torch.as_tensor(
            [1, 0.0, 1.5] if target is None else target
        ).reshape(1, -1)
        self.success_radius = 0.5

    def get_observation(self, indices=None) -> Dict:
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


class HoverEnv2(HoverEnv):
    def __init__(
        self,
        num_agent_per_scene: int = 1,
        num_scene: int = 1,
        seed: int = 42,
        visual: bool = True,
        requires_grad: bool = False,
        random_kwargs: dict = {},
        dynamics_kwargs: dict = {},
        scene_kwargs: dict = {},
        sensor_kwargs: list = [],
        device: str = "cpu",
        target: Optional[torch.Tensor] = None,
        max_episode_steps: int = 256,
        tensor_output: bool = False,
    ):
        sensor_kwargs = [
            {
                "sensor_type": SensorType.DEPTH,
                "uuid": "depth",
                "resolution": [64, 64],
            }
        ]
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
            target=target,
            tensor_output=tensor_output,
        )

    def get_observation(self, indices=None) -> Dict:
        # dis_scale = (
        #     (self.target - self.position)
        #     .norm(dim=1, keepdim=True)
        #     .detach()
        #     .clamp_min(self.max_sense_radius)
        # )
        state = torch.hstack(
            [
                (self.target - self.position) / 10,  # error.
                self.orientation,  # if hovering , target orientation is [1,0,0,0]
                self.velocity / 10,  # target vel is 0
                self.angular_velocity / 10,  # target ang vel is 0
            ]
        ).to(self.device)

        return TensorDict(
            {
                "state": state,
                # "depth": torch.as_tensor(self.sensor_obs["depth"]/10).clamp(max=1)
            }
        )
