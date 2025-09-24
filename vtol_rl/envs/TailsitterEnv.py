from .base.droneGymEnv import DroneGymEnvsBase
from typing import Optional, Dict
import torch
from vtol_rl.utils.type import TensorDict


class DynEnv(DroneGymEnvsBase):
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
            + (self.position - 0).norm(dim=1) * pos_factor
            + (self.orientation - torch.tensor([1, 0, 0, 0])).norm(dim=1) * -0.00001
            + (self.velocity - 0).norm(dim=1) * -0.002
            + (self.angular_velocity - 0).norm(dim=1) * -0.002
        )

        return reward
