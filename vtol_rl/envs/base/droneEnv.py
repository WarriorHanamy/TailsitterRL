import numpy as np
import torch
from dataclasses import dataclass
from vtol_rl.utils.randomization import (
    UniformStateRandomizer,
    load_generator,
)
from vtol_rl.utils.randomization import IMUStateRandomizer

from .dynamics import Dynamics


# REC MARK: kept for future use.
@dataclass
class DroneConfig:
    num_agent_per_scene: int = 1
    num_scene: int = 1
    seed: int = 42
    visual: bool = False
    uav_radius: float = 0.1
    sensitive_radius: float = 10.0
    multi_drone: bool = False
    device: torch.device = torch.device("cpu")


IS_BBOX_COLLISION = True


class DroneEnvsBase:
    def __init__(
        self,
        num_agent_per_scene: int = 1,
        num_scene: int = 1,
        seed: int = 42,
        visual: bool = False,
        random_kwargs: dict | None = {},
        dynamics_kwargs: dict | None = {},
        scene_kwargs: dict | None = {},
        sensor_kwargs: dict | None = {},
        uav_radius: float = 0.1,
        sensitive_radius: float = 10.0,
        multi_drone: bool = False,
        device: type[torch.device] | None = torch.device("cpu"),
    ):
        self.device = device
        self.seed = seed

        self._sensor_obs = {}
        self._is_collision = None
        self._collision_dis = None
        self._collision_point = None
        self._collision_vector = None

        self.uav_radius = uav_radius

        if visual:
            raise ValueError(
                "Habitat-based visual simulation has been removed; set visual=False."
            )
        self.visual = False

        self.noise_settings = random_kwargs.get("noise_kwargs", {})
        self.dynamics = Dynamics(
            num=num_agent_per_scene * num_scene,
            seed=seed,
            device=device,
            drag_random=random_kwargs.get("drag_random", 0.0),
            **dynamics_kwargs,
        )
        self._create_noise_model()

        self.is_multi_drone = multi_drone
        if self.is_multi_drone and (num_agent_per_scene == 1):
            raise ValueError("Num of agents should not be 1 in multi drone env.")

        if "obj_settings" in scene_kwargs:
            scene_kwargs["obj_settings"]["dt"] = self.dynamics.ctrl_dt

        self.stateGenerators = self._create_randomizer(random_kwargs)
        self._scene_iter = random_kwargs.get("scene_iter", False)

        self._create_bbox()
        self._sensor_list = (
            [sensor["uuid"] for sensor in sensor_kwargs]
            if sensor_kwargs is not None
            else []
        )
        self._visual_sensor_list = [s for s in self._sensor_list if "IMU" not in s]

        self._eval = False

    def _create_noise_model(self):
        self.noise_settings["IMU"] = (
            IMUStateRandomizer(
                lin_accl={"mean": [0.0, 0.0, 0.0], "std": [0.002, 0.002, 0.002]},
                ang_vel={"mean": [0.0, 0.0, 0.0], "std": [0.001, 0.001, 0.001]},
            ).to(self.device),
        )

    def _generate_noise_obs(self, sensor):
        if sensor == "IMU":
            lin_accl, ang_vel = self.noise_settings["IMU"][0]._generate(
                self.dynamics.num
            )
            state_with_noise = torch.cat(
                [
                    self.state[:, :10],
                    self.state[:, 10:13] + ang_vel,
                    self.state[:, 13:],
                ],
                dim=1,
            )
            # normalize the orientation
            normalized_ori = torch.nn.functional.normalize(
                state_with_noise[:, 3:7], p=2, dim=1
            )
            state_with_noise = torch.cat(
                [state_with_noise[:, :3], normalized_ori, state_with_noise[:, 7:]],
                dim=1,
            )
            return state_with_noise

    def _create_bbox(self):
        if not self.visual:
            bboxes = [
                torch.tensor([[-30.0, -20.0, 0.0], [30.0, 20.0, 8.0]]).to(self.device)
            ]
            self._bboxes = bboxes
            self._flatten_bboxes = [bbox.flatten() for bbox in bboxes]

    def _create_randomizer(self, random_kwargs: dict):
        default_state_generator_config = {
            "class": UniformStateRandomizer,
            "kwargs": {
                "position": {"mean": [0.0, 0.0, 1.0], "radius": [0.0, 0.0, 0.1]},
            },
        }
        state_generator_config = default_state_generator_config | random_kwargs.get(
            "state_generator", {}
        )

        stateGenerators = []
        if self.visual:
            pass
        else:
            for agent_id in range(self.dynamics.num):
                stateGenerators.append(
                    load_generator(
                        cls=state_generator_config["class"],
                        device=self.device,
                        kwargs=state_generator_config["kwargs"],
                    )
                )
        return stateGenerators

    def _generate_state(
        self, indices: list[int] | None = None
    ) -> tuple[torch.Tensor, ...]:
        """
        Generate random state for the agents.
        Args:
            indices (list[int] | None): indices of agents to generate state for. If None, generate for all agents.
        Returns:
            states components: (position, orientation, velocity, angular_velocity)
            with shapes (N, 3), (N, 4), (N, 3), (N, 3), respectively. N = len(indices)
        """
        indices = np.arange(self.dynamics.num) if indices is None else indices
        indices = (
            torch.as_tensor([indices], device=self.device)
            if not hasattr(indices, "__iter__")
            else indices
        )
        positions, orientations, velocities, angular_velocities = (
            torch.empty((len(indices), 3), device=self.device),
            torch.empty((len(indices), 4), device=self.device),
            torch.empty((len(indices), 3), device=self.device),
            torch.empty((len(indices), 3), device=self.device),
        )
        for index_idx, index_val in enumerate(indices):
            (
                positions[index_idx],
                orientations[index_idx],
                velocities[index_idx],
                angular_velocities[index_idx],
            ) = self.stateGenerators[index_idx].generate(num=1)

        return positions, orientations, velocities, angular_velocities

    def reset(self, state=None) -> tuple[torch.Tensor, np.ndarray | None]:
        self.reset_agents(indices=None, state=state)
        return self.state, self.sensor_obs

    def reset_agents(
        self, indices: list | None = None, state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, np.ndarray | None]:
        """
        Reset the agents to a specific state or random state.
        If state is None, random state will be generated.
        If indices is None, all agents will be reset.
        Args:
            indices (list | None): indices of agents to reset.
            state (torch.Tensor | None): specific state to reset the agents to.
        Returns:
            update self.state and self.sensor_obs
        """
        indices = (
            indices
            if (indices is None or hasattr(indices, "__iter__"))
            else torch.as_tensor([indices], device=self.device)
        )

        motor_speed, thrust, t = None, None, None
        if state is not None:
            if isinstance(state, torch.Tensor):
                state = state.to(self.device)
                (
                    pos,
                    ori,
                    vel,
                    ori_vel,
                    motor_speed,
                    thrust,
                    t,
                ) = torch.split(state.clone().detach(), [3, 4, 3, 3, 4, 4, 1], dim=1)
        else:
            pos, ori, vel, ori_vel = self._generate_state(indices)

        self.dynamics.reset(
            pos=pos,
            ori=ori,
            vel=vel,
            ori_vel=ori_vel,
            motor_omega=motor_speed,
            thrusts=thrust,
            t=t,
            indices=indices,
        )
        self.update_observation(indices=indices)

    def update_observation(self, indices=None):
        self._sensor_obs["IMU"] = self._generate_noise_obs("IMU")

    def update_collision(self, indices: list[int] | None = None):
        if indices is None:
            value, index = torch.hstack(
                [
                    self.dynamics.position.clone().detach() - self._bboxes[0][0],
                    self._bboxes[0][1] - self.dynamics.position.clone().detach(),
                ]
            ).min(dim=1)
            self._collision_point = self.dynamics.position.clone().detach()
            self._collision_point[torch.arange(self.dynamics.num), index % 3] = (
                self._flatten_bboxes[0][index]
            )
        else:
            value, index = torch.hstack(
                [
                    self.dynamics.position[indices].clone().detach()
                    - self._bboxes[0][0],
                    self._bboxes[0][1]
                    - self.dynamics.position[indices].clone().detach(),
                ]
            ).min(dim=1)
            self._collision_point[indices] = (
                self.dynamics.position[indices].clone().detach()
            )
            self._collision_point[indices, index % 3] = self._flatten_bboxes[0][index]

        self._is_out_bounds = (self.dynamics.position < self._bboxes[0][0]).any(
            dim=1
        ) | (self.dynamics.position > self._bboxes[0][1]).any(dim=1)

        self._collision_vector = self._collision_point - self.position
        self._collision_dis = (self._collision_vector - 0).norm(dim=1)
        self._is_collision = (
            self._collision_dis < self.uav_radius
        ) | self._is_out_bounds

    def step(self, action):
        if isinstance(action, torch.Tensor):
            if action.dim() == 3 and action.size(-1) == 1:
                action = action.squeeze(-1)
        self.dynamics.step(action)
        self.update_observation()
        self.update_collision()

    def set_seed(self, seed: int | None = 42):
        seed = self.seed if seed is None else seed
        self.dynamics.set_seed(seed)

    def stack(self):
        self._stack_cache = (
            self.position.clone().detach(),
            self.orientation.clone().detach(),
            self.velocity.clone().detach(),
            self.angular_velocity.clone().detach(),
        )

    def recover(self):
        self.reset_agents(state=self._stack_cache)

    def detach(self):
        self.dynamics.detach()

    def close(self):
        self.dynamics.close()

    def render(self, **kwargs):
        raise ValueError("Visual rendering is no longer supported without habitat_sim.")

    @property
    def state(self):
        """
        Returns:
            torch.Tensor: shape is (num_agents, state_dim), currently (num_agents, 13)
        """
        return self.dynamics.state

    @property
    def sensor_obs(self):
        """
        Returns:
            dict of torch.Tensor: each tensor shape is (num_agents, sensor_dim)
        """
        return self._sensor_obs

    @property
    def is_collision(self):
        return self._is_collision

    # @property
    # def closest_obstacle_dis(self):
    #     return self._collision_dis

    @property
    def direction(self):
        return self.dynamics.direction

    @property
    def position(self):
        return self.dynamics.position

    @property
    def orientation(self):
        return self.dynamics.orientation

    @property
    def velocity(self):
        return self.dynamics.velocity

    @property
    def angular_velocity(self):
        return self.dynamics.angular_velocity

    @property
    def t(self):
        return self.dynamics.t

    @property
    def thrusts(self):
        return self.dynamics.thrusts

    @property
    def full_state(self):
        return self.dynamics.full_state

    @property
    def acceleration(self):
        return self.dynamics.acceleration

    @property
    def angular_acceleration(self):
        return self.dynamics.angular_acceleration

    @property
    def collision_point(self):
        return self._collision_point

    @property
    def collision_vector(self):
        return self._collision_vector

    @property
    def collision_dis(self):
        return self._collision_dis

    @property
    def dynamic_object_position(self):
        return [[None] for _ in range(self.dynamics.num)]

    @property
    def dynamic_object_velocity(self):
        return [[None] for _ in range(self.dynamics.num)]

    @property
    def dynamic_object_acceleration(self):
        return [[None] for _ in range(self.dynamics.num)]
