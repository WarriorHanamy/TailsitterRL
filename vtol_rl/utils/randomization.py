import torch
from .type import Uniform, Gaussian
from typing import Optional
from abc import abstractmethod
from typing import TypeVar, Type

rotation_matrices = torch.tensor(
    [
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],  # 0°
        [[0, -1, 0], [1, 0, 0], [0, 0, 1]],  # 90°
        [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],  # 180°
        [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],  # 270°
    ],
    dtype=torch.float32,
)


class StateRandomizer:
    def __init__(
        self,
        position: dict | None = None,
        orientation: dict | None = None,  # euler angle
        velocity: dict | None = None,
        angular_velocity: dict | None = None,
        seed: int = 42,
        is_collision_func: Optional[callable] = None,
        scene_id: Optional[int] = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.position = position
        self.orientation = orientation
        self.velocity = velocity
        self.angular_velocity = angular_velocity
        self.is_collision_func = is_collision_func
        self.device = device
        self.set_seed(seed)

    @abstractmethod
    def _generate(self, num) -> tuple:
        pass

    def generate(self, num, _eval=False):
        raw_pos, raw_ori, raw_vel, raw_ang_vel = self._generate(num)
        return raw_pos, raw_ori, raw_vel, raw_ang_vel

    def set_seed(self, seed=42):
        torch.manual_seed(seed)

    def to(self, device):
        self.device = device
        return self


class UniformStateRandomizer(StateRandomizer):
    def __init__(
        self,
        position={"mean": [0.0, 0.0, 0.0], "radius": [0.0, 0.0, 0.0]},
        orientation={"mean": [0.0, 0.0, 0.0], "radius": [0.0, 0.0, 0.0]},  # euler angle
        velocity={"mean": [0.0, 0.0, 0.0], "radius": [0.0, 0.0, 0.0]},
        angular_velocity={"mean": [0.0, 0.0, 0.0], "radius": [0.0, 0.0, 0.0]},
        seed: int = 42,
        is_collision_func: Optional[callable] = None,
        scene_id: Optional[int] = None,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(
            position=position,
            orientation=orientation,
            velocity=velocity,
            angular_velocity=angular_velocity,
            seed=seed,
            is_collision_func=is_collision_func,
            scene_id=scene_id,
            device=device,
        )

        self.position_randomizer = Uniform(**position).to(self.device)
        self.orientation_randomizer = Uniform(**orientation).to(self.device)
        self.velocity_randomizer = Uniform(**velocity).to(self.device)
        self.angular_velocity_randomizer = Uniform(**angular_velocity).to(self.device)

    def _generate(self, num) -> tuple:
        position = self.position_randomizer.sample(num).to(self.device)
        orientation = self.orientation_randomizer.sample(num).to(self.device)
        velocity = self.velocity_randomizer.sample(num).to(self.device)
        angular_velocity = self.angular_velocity_randomizer.sample(num).to(self.device)
        return position, orientation, velocity, angular_velocity


class GaussianStateRandomizer(StateRandomizer):
    def __init__(
        self,
        position={"mean": [0.0, 0.0, 0.0], "std": [0.0, 0.0, 0.0]},
        orientation={"mean": [0.0, 0.0, 0.0], "std": [0.0, 0.0, 0.0]},  # euler angle
        velocity={"mean": [0.0, 0.0, 0.0], "std": [0.0, 0.0, 0.0]},
        angular_velocity={"mean": [0.0, 0.0, 0.0], "std": [0.0, 0.0, 0.0]},
        seed: int = 42,
        is_collision_func: Optional[callable] = None,
        scene_id: Optional[int] = None,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(
            position=position,
            orientation=orientation,
            velocity=velocity,
            angular_velocity=angular_velocity,
            seed=seed,
            is_collision_func=is_collision_func,
            scene_id=scene_id,
            device=device,
        )

        self.position_randomizer = Gaussian(**position).to(self.device)
        self.orientation_randomizer = Gaussian(**orientation).to(self.device)
        self.velocity_randomizer = Gaussian(**velocity).to(self.device)
        self.angular_velocity_randomizer = Gaussian(**angular_velocity).to(self.device)

    def _generate(self, num) -> tuple:
        position = self.position_randomizer.sample(num).to(self.device)
        orientation = self.orientation_randomizer.sample(num).to(self.device)
        velocity = self.velocity_randomizer.sample(num).to(self.device)
        angular_velocity = self.angular_velocity_randomizer.sample(num).to(self.device)
        return position, orientation, velocity, angular_velocity


class IMUStateRandomizer(StateRandomizer):
    def __init__(
        self,
        lin_accl: dict = {"mean": [0.0, 0.0, 0.0], "std": [0.0, 0.0, 0.0]},
        ang_vel: dict = {"mean": [0.0, 0.0, 0.0], "std": [0.0, 0.0, 0.0]},
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(device=device)

        self.lin_accl_randomizer = Gaussian(**lin_accl).to(self.device)
        self.ang_vel_randomizer = Gaussian(**ang_vel).to(self.device)

    def _generate(self, num) -> tuple:
        lin_accl = self.lin_accl_randomizer.sample(num).to(self.device)
        ang_vel = self.ang_vel_randomizer.sample(num).to(self.device)
        return lin_accl, ang_vel


class TargetUniformRandomizer(UniformStateRandomizer):
    def __init__(self, min_dis=0.5, max_dis=10.0, test=False, *args, **kwargs):
        self.min_dis = min_dis
        self.max_dis = max_dis
        self.test = test
        if self.test:
            self.current_generate_index = 0
        super().__init__(*args, **kwargs)

    def _generate(self, num, **kwargs) -> tuple:
        def calculate_yaw_pitch(vector):
            """s
            Calculate the yaw and pitch angles of a vector.

            Args:
                vector (np.ndarray): A 3D vector [x, y, z].

            Returns:
                tuple: (yaw, pitch) in radians.
            """
            x, y, z = vector[:, 0], vector[:, 1], vector[:, 2]

            # Calculate yaw (arctan2 handles the quadrant correctly)
            yaw = torch.arctan2(y, x)
            yaw = torch.arccos(x / vector[:, :2].norm(dim=1)) * y.sign()
            # Calculate pitch
            norm = torch.linalg.norm(vector)  # Magnitude of the vector
            pitch = torch.arcsin(z / norm)
            return yaw, pitch

        target_position = kwargs["position"]
        if not self.test:
            position = (
                2 * torch.rand(num, *self.position.radius.shape) - 1
            ) * self.position.radius.unsqueeze(0)
        else:
            position = torch.tile(kwargs["velocity"].unsqueeze(0), (num, 1))
            position = (
                rotation_matrices[self.current_generate_index % 4] @ position.T
            ).T
            self.current_generate_index += 1

        position_norm = position.norm(dim=1, keepdim=True)
        # Create scaling factor
        scale_factor = torch.ones_like(position_norm)
        # If norm > max_dis, scale down
        scale_factor = torch.where(
            position_norm > self.max_dis, self.max_dis / position_norm, scale_factor
        )
        # If norm < min_dis, scale up
        scale_factor = torch.where(
            position_norm < self.min_dis, self.min_dis / position_norm, scale_factor
        )
        # Apply scaling
        position = position * scale_factor
        position = position + target_position.unsqueeze(0)
        direction = target_position.unsqueeze(0) - position
        yaw, pitch = calculate_yaw_pitch(direction)
        orientation = (
            torch.stack([torch.zeros(num), pitch * 0, yaw], dim=1)
            + (2 * torch.rand(num, 3) - 1) * self.orientation.radius
        )  # yaw, pitch, roll
        if "velocity" in kwargs.keys():
            velocity = torch.tile(kwargs["velocity"].unsqueeze(0), (num, 1))
        else:
            velocity = (
                2 * torch.rand(num, 3) - 1
            ) * self.velocity.radius + self.velocity.mean
        angular_velocity = (
            2 * torch.rand(num, 3) - 1
        ) * self.angular_velocity.radius + self.angular_velocity.mean

        return position, orientation, velocity, angular_velocity


class UnionRandomizer:
    Randomizer_alias = {
        "Uniform": UniformStateRandomizer,
        "Normal": GaussianStateRandomizer,
    }

    def __init__(
        self,
        randomizers_kwargs: list,
        device,
        is_collision_func=None,
        scene_id=0,
    ):
        self.randomizers = []
        for randomizers in randomizers_kwargs:
            self.randomizers.append(
                self.Randomizer_alias[randomizers["class"]](
                    device=device,
                    is_collision_func=is_collision_func,
                    scene_id=scene_id,
                    **randomizers["kwargs"],
                )
            )

    def __Len__(self):
        return len(self.randomizers)

    def to(self, device):
        for randomizer in self.randomizers:
            randomizer.to(device)

    def _generate(self, num) -> tuple:
        position, orientation, velocity, angular_velocity = [], [], [], []
        for randomizer in self.randomizers:
            pos, ori, vel, ang_vel = randomizer.generate(num)
            position.append(pos)
            orientation.append(ori)
            velocity.append(vel)
            angular_velocity.append(ang_vel)

        position, orientation, velocity, angular_velocity = (
            torch.stack(position),
            torch.stack(orientation),
            torch.stack(velocity),
            torch.stack(angular_velocity),
        )
        select_randomizer_index = torch.randint(0, len(self.randomizers), (num,))
        row = torch.arange(num)
        return (
            position[row, select_randomizer_index],
            orientation[row, select_randomizer_index],
            velocity[row, select_randomizer_index],
            angular_velocity[row, select_randomizer_index],
        )

    def safe_generate(self, num) -> tuple:
        position, orientation, velocity, angular_velocity = [], [], [], []
        for randomizer in self.randomizers:
            pos, ori, vel, ang_vel = randomizer.safe_generate(num)
            position.append(pos)
            orientation.append(ori)
            velocity.append(vel)
            angular_velocity.append(ang_vel)

        position, orientation, velocity, angular_velocity = (
            torch.stack(position),
            torch.stack(orientation),
            torch.stack(velocity),
            torch.stack(angular_velocity),
        )
        select_randomizer_index = torch.randint(0, len(self.randomizers), (num,))
        row = torch.arange(num)
        return (
            position[select_randomizer_index, row, :],
            orientation[select_randomizer_index, row, :],
            velocity[select_randomizer_index, row, :],
            angular_velocity[select_randomizer_index, row, :],
        )


RandomizerType = TypeVar("RandomizerType", bound="StateRandomizer")


def load_generator(
    cls: Type[RandomizerType],
    kwargs,
    is_collision_func=None,
    scene_id=None,
    device="cpu",
):
    return cls(
        is_collision_func=is_collision_func, scene_id=scene_id, device=device, **kwargs
    )


# def load_dist(data):
#     cls_alias = {
#         "Uniform": Uniform,
#         "Normal": Gaussian,
#     }
#     if not isinstance(data, dict):
#         kwargs = {"mean": data, "half": 0.0}
#         cls = Uniform
#     else:
#         cls = cls_alias[data["class"]]
#         kwargs = data["kwargs"]
#     return cls(**kwargs)
