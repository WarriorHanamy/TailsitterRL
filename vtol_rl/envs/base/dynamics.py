from __future__ import annotations

from collections.abc import Iterable

import torch

from vtol_rl import config
from vtol_rl.utils.maths import Integrator, Quaternion
from vtol_rl.utils.type import ACTION_TYPE, Uniform, action_type_alias, bound


class Dynamics:
    """
    Closed-loop VTOL dynamics model with thrust and angular-rate actuators.
    """

    g = torch.tensor([[0.0, 0.0, -9.81]], dtype=torch.float32)
    z = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)

    def __init__(
        self,
        num: int = 1,
        action_type: str = "bodyrate",
        ori_output_type: str = "quaternion",
        seed: int = 42,
        sim_time_step: float = 0.005,
        ctrl_period: float = 0.03,
        ctrl_delay: bool = True,
        comm_delay: float = 0.06,
        action_space: tuple[float, float] = (-1, 1),
        device: torch.device | str = torch.device("cpu"),
        integrator: str = "euler",
        drag_random: float = 0.0,
        cfg: str = "drone_state",
    ):
        if action_type not in ("bodyrate",):
            raise ValueError("Only 'bodyrate' action type is supported.")
        if ori_output_type not in ("quaternion",):
            raise ValueError("Only quaternion orientation output is supported.")

        self.num = int(num)
        self.device = torch.device(device)
        self.dtype = torch.float32

        self.action_type = action_type_alias[action_type]
        self.ori_output_type = ori_output_type

        self.sim_time_step = float(sim_time_step)
        self.ctrl_period = float(ctrl_period)
        if self.ctrl_period < self.sim_time_step:
            raise ValueError(
                "ctrl_period must be >= sim_time_step; set ctrl_period as a multiple of sim_time_step."
            )
        ratio = torch.as_tensor(self.ctrl_period) / torch.as_tensor(self.sim_time_step)
        if torch.any(ratio % 1 != 0):
            raise ValueError("ctrl_period should be an integer multiple of sim_time_step.")

        self._interval_steps = int(round(self.ctrl_period / self.sim_time_step))
        self._comm_delay_steps = int(comm_delay / self.ctrl_period)
        self._ctrl_delay = bool(ctrl_delay)
        self._integrator = "1st_order_euler" if integrator == "euler" else integrator

        self.cfg = cfg
        self._drag_random = float(drag_random)
        self.action_space = action_space

        self.set_seed(seed)

        self._load_parameters()
        self._build_closed_loop_constants()
        self._init_state_tensors()

        self._pre_action = [
            torch.zeros((self.num, 4), device=self.device, dtype=self.dtype)
            for _ in range(self._comm_delay_steps)
        ]

        self._get_scale_factor()

    # --------------------------------------------------------------------- #
    # Initialization helpers
    # --------------------------------------------------------------------- #
    def _load_parameters(self) -> None:
        data = config.load_json(f"drone/{self.cfg}.json")

        self.name = data["name"]

        self.m = torch.tensor(data["mass"], dtype=self.dtype, device=self.device)
        inertia_diag = torch.as_tensor(
            data["inertia"], dtype=self.dtype, device=self.device
        )
        self._inertia = torch.diag(inertia_diag)
        self._inertia_inv = torch.inverse(self._inertia)

        cross_sections = torch.as_tensor(
            data["cross_sections"], dtype=self.dtype, device=self.device
        ).view(1, -1)
        quad_drag = torch.as_tensor(
            data["quad_drag_coeffs"], dtype=self.dtype, device=self.device
        ).view(1, -1)
        self._quad_drag_coeffs_mean = 0.5 * 1.225 * cross_sections * quad_drag
        self._linear_drag_coeffs_mean = torch.as_tensor(
            data["linear_drag_coeffs"], dtype=self.dtype, device=self.device
        ).view(1, -1)

        self._rotor_speed_to_thrust_coefs = torch.as_tensor(
            data["rotor_speed_to_thrust_coefs"], dtype=self.dtype, device=self.device
        )

        motor_max = torch.tensor(data["motor_omega_max"], dtype=self.dtype, device=self.device)
        motor_min = torch.tensor(data["motor_omega_min"], dtype=self.dtype, device=self.device)
        self._bd_rotor_omega = bound(min=motor_min.item(), max=motor_max.item())

        thrust_max = (
            self._rotor_speed_to_thrust_coefs[0] * motor_max.pow(2)
            + self._rotor_speed_to_thrust_coefs[1] * motor_max
            + self._rotor_speed_to_thrust_coefs[2]
        )
        thrust_min = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        self._bd_thrust = bound(min=thrust_min.item(), max=thrust_max.item())

        max_rate = torch.tensor(data["max_rate"], dtype=self.dtype, device=self.device)
        self._bd_rate = bound(min=(-max_rate).item(), max=max_rate.item())

        max_spd = torch.tensor(data["max_spd"], dtype=self.dtype, device=self.device)
        self._bd_spd = bound(min=(-max_spd).item(), max=max_spd.item())

        max_pos = torch.tensor(data["max_pos"], dtype=self.dtype, device=self.device)
        self._bd_pos = bound(min=(-max_pos).item(), max=max_pos.item())

    def _build_closed_loop_constants(self) -> None:
        self._gravity = Dynamics.g.to(device=self.device, dtype=self.dtype)
        self._body_z = Dynamics.z.to(device=self.device, dtype=self.dtype)

        self._hover_total_thrust = self.m * -self._gravity[0, 2]
        self._hover_thrust_per_rotor = self._hover_total_thrust / 4
        self._max_total_thrust = self._bd_thrust.max * 4
        self._init_thrust_mag = self._hover_thrust_per_rotor.item()
        hover_thrusts = torch.full(
            (1, 4),
            self._init_thrust_mag,
            device=self.device,
            dtype=self.dtype,
        )
        self._init_motor_omega = self._compute_rotor_omega(hover_thrusts)[0, 0].item()

        self._thrust_time_constant = torch.tensor(0.05, device=self.device, dtype=self.dtype)

        zeta_xy = torch.tensor([0.7, 0.7], device=self.device, dtype=self.dtype)
        omega_n_xy = torch.tensor([100.0, 100.0], device=self.device, dtype=self.dtype)
        zeta_z = torch.tensor([1.5], device=self.device, dtype=self.dtype)
        omega_n_z = torch.tensor([8.0], device=self.device, dtype=self.dtype)
        self._zeta = torch.cat([zeta_xy, zeta_z]).view(1, 3)
        self._omega_n = torch.cat([omega_n_xy, omega_n_z]).view(1, 3)

        self._acc_noise_std = torch.tensor(0.5, device=self.device, dtype=self.dtype)

    def _init_state_tensors(self) -> None:
        self._position = torch.zeros((self.num, 3), device=self.device, dtype=self.dtype)
        self._orientation = Quaternion(num=self.num, device=self.device)
        self._velocity = torch.zeros((self.num, 3), device=self.device, dtype=self.dtype)
        self._angular_velocity = torch.zeros((self.num, 3), device=self.device, dtype=self.dtype)

        self._thrust_state = torch.full(
            (self.num, 1),
            self._hover_total_thrust.item(),
            device=self.device,
            dtype=self.dtype,
        )
        self._angular_velocity_state = torch.zeros(
            (self.num, 3), device=self.device, dtype=self.dtype
        )

        self._angular_acc = torch.zeros((self.num, 3), device=self.device, dtype=self.dtype)
        self._angular_jerk = torch.zeros((self.num, 3), device=self.device, dtype=self.dtype)
        self._acc = torch.zeros((self.num, 3), device=self.device, dtype=self.dtype)
        self._t = torch.zeros((self.num,), device=self.device, dtype=self.dtype)

        self._linear_drag_coeffs = self._linear_drag_coeffs_mean.repeat(self.num, 1)
        self._quad_drag_coeffs = self._quad_drag_coeffs_mean.repeat(self.num, 1)

        self._thrusts = self._thrust_state.repeat(1, 4) / 4
        self._motor_omega = self._compute_rotor_omega(self._thrusts)

    # --------------------------------------------------------------------- #
    # Reset and state manipulation
    # --------------------------------------------------------------------- #
    def _reset_batch_size(self, indices) -> int:
        if indices is None:
            return self.num
        if isinstance(indices, torch.Tensor):
            return int(indices.numel())
        if isinstance(indices, Iterable):
            return len(list(indices))
        raise TypeError("indices must be None, Tensor, or iterable of indices.")

    def _coerce_reset_tensor(
        self,
        name: str,
        value,
        expected_feature_dim: int,
        batch_size: int,
    ) -> torch.Tensor | None:
        if value is None:
            return None

        if isinstance(value, torch.Tensor):
            tensor = value.to(device=self.device, dtype=self.dtype)
        else:
            tensor = torch.as_tensor(value, dtype=self.dtype, device=self.device)

        if tensor.ndim == 1 and expected_feature_dim == 1:
            tensor = tensor.view(-1, 1)

        if tensor.ndim != 2:
            raise ValueError(
                f"{name} must have shape (batch, {expected_feature_dim}); got tensor with {tensor.ndim} dims"
            )

        if tensor.shape[-1] != expected_feature_dim:
            raise ValueError(
                f"{name} must have shape (batch, {expected_feature_dim}); received {tuple(tensor.shape)}"
            )

        if tensor.shape[0] != batch_size:
            raise ValueError(
                f"{name} batch dimension must match number of indices ({batch_size}); received {tuple(tensor.shape)}"
            )

        return tensor.contiguous()

    def _coerce_reset_times(self, value, batch_size: int) -> torch.Tensor | None:
        if value is None:
            return None

        if isinstance(value, torch.Tensor):
            tensor = value.to(device=self.device, dtype=self.dtype)
        else:
            tensor = torch.as_tensor(value, dtype=self.dtype, device=self.device)

        if tensor.ndim == 1:
            if tensor.shape[0] != batch_size:
                raise ValueError(
                    f"t must have shape (batch,) or (batch, 1); received {tuple(tensor.shape)}"
                )
            return tensor

        if tensor.ndim == 2:
            if tensor.shape[0] != batch_size or tensor.shape[1] != 1:
                raise ValueError(
                    f"t must have shape (batch,) or (batch, 1); received {tuple(tensor.shape)}"
                )
            return tensor.squeeze(1)

        raise ValueError("t must be a 1D or 2D tensor")

    def reset(
        self,
        pos: list | torch.Tensor | None = None,
        ori: list | torch.Tensor | None = None,
        vel: list | torch.Tensor | None = None,
        ori_vel: list | torch.Tensor | None = None,
        motor_omega: list | torch.Tensor | None = None,
        thrusts: list | torch.Tensor | None = None,
        thrust_state: list | torch.Tensor | None = None,
        ang_vel_state: list | torch.Tensor | None = None,
        t: list | torch.Tensor | None = None,
        indices: list | torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = self._reset_batch_size(indices)
        pos_tensor = self._coerce_reset_tensor("pos", pos, 3, batch_size)
        vel_tensor = self._coerce_reset_tensor("vel", vel, 3, batch_size)
        ori_vel_tensor = self._coerce_reset_tensor("ori_vel", ori_vel, 3, batch_size)
        ori_tensor = self._coerce_reset_tensor("ori", ori, 4, batch_size)
        thrusts_tensor = self._coerce_reset_tensor("thrusts", thrusts, 4, batch_size)
        motor_tensor = self._coerce_reset_tensor("motor_omega", motor_omega, 4, batch_size)
        thrust_state_tensor = self._coerce_reset_tensor("thrust_state", thrust_state, 1, batch_size)
        ang_vel_state_tensor = self._coerce_reset_tensor("ang_vel_state", ang_vel_state, 3, batch_size)
        time_tensor = self._coerce_reset_times(t, batch_size)

        if indices is None:
            self._position = (
                torch.zeros((self.num, 3), device=self.device, dtype=self.dtype)
                if pos_tensor is None
                else pos_tensor
            )
            self._orientation = (
                Quaternion(num=self.num, device=self.device)
                if ori_tensor is None
                else Quaternion(
                    ori_tensor[:, 0],
                    ori_tensor[:, 1],
                    ori_tensor[:, 2],
                    ori_tensor[:, 3],
                )
            )
            self._velocity = (
                torch.zeros((self.num, 3), device=self.device, dtype=self.dtype)
                if vel_tensor is None
                else vel_tensor
            )
            self._angular_velocity = (
                torch.zeros((self.num, 3), device=self.device, dtype=self.dtype)
                if ori_vel_tensor is None
                else ori_vel_tensor
            )
            self._thrust_state = (
                torch.full(
                    (self.num, 1),
                    self._hover_total_thrust.item(),
                    device=self.device,
                    dtype=self.dtype,
                )
                if thrust_state_tensor is None
                else thrust_state_tensor
            )
            self._angular_velocity_state = (
                torch.zeros((self.num, 3), device=self.device, dtype=self.dtype)
                if ang_vel_state_tensor is None
                else ang_vel_state_tensor
            )
            self._t = (
                torch.zeros((self.num,), device=self.device, dtype=self.dtype)
                if time_tensor is None
                else time_tensor
            )
        else:
            idx = (
                indices
                if isinstance(indices, slice)
                else torch.as_tensor(indices, device=self.device, dtype=torch.long)
            )
            self._position[idx] = (
                torch.zeros((batch_size, 3), device=self.device, dtype=self.dtype)
                if pos_tensor is None
                else pos_tensor
            )
            if ori_tensor is None:
                self._orientation[idx] = Quaternion(
                    num=batch_size, device=self.device
                )
            else:
                self._orientation[idx] = ori_tensor
            self._velocity[idx] = (
                torch.zeros((batch_size, 3), device=self.device, dtype=self.dtype)
                if vel_tensor is None
                else vel_tensor
            )
            self._angular_velocity[idx] = (
                torch.zeros((batch_size, 3), device=self.device, dtype=self.dtype)
                if ori_vel_tensor is None
                else ori_vel_tensor
            )
            self._thrust_state[idx] = (
                torch.full(
                    (batch_size, 1),
                    self._hover_total_thrust.item(),
                    device=self.device,
                    dtype=self.dtype,
                )
                if thrust_state_tensor is None
                else thrust_state_tensor
            )
            self._angular_velocity_state[idx] = (
                torch.zeros((batch_size, 3), device=self.device, dtype=self.dtype)
                if ang_vel_state_tensor is None
                else ang_vel_state_tensor
            )
            if time_tensor is None:
                self._t[idx] = torch.zeros(
                    (batch_size,), device=self.device, dtype=self.dtype
                )
            else:
                self._t[idx] = time_tensor

        self._angular_acc.zero_()
        self._angular_jerk.zero_()
        self._acc.zero_()

        self._thrusts = (
            self._thrust_state.repeat(1, 4) / 4
            if thrusts_tensor is None
            else thrusts_tensor
        )
        self._motor_omega = (
            self._compute_rotor_omega(self._thrusts)
            if motor_tensor is None
            else motor_tensor
        )

        self._linear_drag_coeffs = self._linear_drag_coeffs_mean.repeat(self.num, 1)
        self._quad_drag_coeffs = self._quad_drag_coeffs_mean.repeat(self.num, 1)
        self._apply_drag_randomization(indices)

        self._pre_action = [
            torch.zeros((self.num, 4), device=self.device, dtype=self.dtype)
            for _ in range(self._comm_delay_steps)
        ]

        return self.state

    def _apply_drag_randomization(self, indices) -> None:
        if self._drag_random <= 0:
            return

        def _rand_like(tensor: torch.Tensor) -> torch.Tensor:
            return (
                (torch.rand_like(tensor) - 0.5) * 2 * self._drag_random
            ).clamp(-0.5, 0.5) + 1.0

        if indices is None:
            self._linear_drag_coeffs = self._linear_drag_coeffs_mean.repeat(self.num, 1) * _rand_like(
                self._linear_drag_coeffs_mean.repeat(self.num, 1)
            )
            self._quad_drag_coeffs = self._quad_drag_coeffs_mean.repeat(self.num, 1) * _rand_like(
                self._quad_drag_coeffs_mean.repeat(self.num, 1)
            )
        else:
            idx = (
                indices
                if isinstance(indices, slice)
                else torch.as_tensor(indices, device=self.device, dtype=torch.long)
            )
            self._linear_drag_coeffs[idx] = (
                self._linear_drag_coeffs_mean.repeat(self.num, 1)[idx] * _rand_like(self._linear_drag_coeffs_mean.repeat(self.num, 1)[idx])
            )
            self._quad_drag_coeffs[idx] = (
                self._quad_drag_coeffs_mean.repeat(self.num, 1)[idx] * _rand_like(self._quad_drag_coeffs_mean.repeat(self.num, 1)[idx])
            )

    def detach(self) -> None:
        self._position = self._position.clone().detach()
        self._orientation = self._orientation.clone().detach()
        self._velocity = self._velocity.clone().detach()
        self._angular_velocity = self._angular_velocity.clone().detach()
        self._angular_velocity_state = self._angular_velocity_state.clone().detach()
        self._angular_acc = self._angular_acc.clone().detach()
        self._angular_jerk = self._angular_jerk.clone().detach()
        self._thrust_state = self._thrust_state.clone().detach()
        self._acc = self._acc.clone().detach()
        self._t = self._t.clone().detach()
        self._thrusts = self._thrusts.clone().detach()
        self._motor_omega = self._motor_omega.clone().detach()
        self._pre_action = [pre.clone().detach() for pre in self._pre_action]

    # --------------------------------------------------------------------- #
    # Simulation step
    # --------------------------------------------------------------------- #
    def step(self, action: torch.Tensor) -> torch.Tensor:
        if action.ndim != 2 or action.shape[1] != 4:
            raise ValueError("action must have shape (num_envs, 4)")
        if action.shape[0] != self.num:
            raise ValueError(f"Expected {self.num} actions, got {action.shape[0]}")

        if self._comm_delay_steps:
            self._pre_action.append(action.clone())
            action = self._pre_action.pop(0)

        command = self._de_normalize(action)
        target_thrust = (command[:, :1] * self.m).clamp(min=0.0, max=self._max_total_thrust)
        target_bodyrate = command[:, 1:]

        for _ in range(self._interval_steps):
            self._update_thrust_state(target_thrust, self.sim_time_step)
            self._update_angular_state(target_bodyrate, self.sim_time_step)
            self._update_linear_dynamics(self.sim_time_step)

        self._t += self.ctrl_period
        self._ugly_fix()
        return self.state

    def _update_thrust_state(self, target_thrust: torch.Tensor, dt: float) -> None:
        thrust_error = target_thrust - self._thrust_state
        thrust_derivative = thrust_error / self._thrust_time_constant
        self._thrust_state = (self._thrust_state + thrust_derivative * dt).clamp(
            min=0.0, max=self._max_total_thrust
        )
        self._thrusts = self._thrust_state.repeat(1, 4) / 4
        self._motor_omega = self._compute_rotor_omega(self._thrusts)

    def _update_angular_state(self, target_bodyrate: torch.Tensor, dt: float) -> None:
        omega_error = target_bodyrate - self._angular_velocity
        omega_n_sq = self._omega_n.pow(2)
        damping = 2 * self._zeta * self._omega_n * self._angular_velocity_state
        angular_jerk = omega_n_sq * omega_error - damping

        self._angular_velocity_state = self._angular_velocity_state + angular_jerk * dt
        self._angular_velocity = self._angular_velocity + self._angular_velocity_state * dt

        self._angular_jerk = angular_jerk
        self._angular_acc = self._angular_velocity_state

        self._angular_velocity = self._angular_velocity.clamp(
            min=self._bd_rate.min, max=self._bd_rate.max
        )

    def _update_linear_dynamics(self, dt: float) -> None:
        thrust_body = self._body_z * self._thrust_state
        velocity_body = self._orientation.inv_rotate(self._velocity)

        linear_drag = self._linear_drag_coeffs * velocity_body
        quadratic_drag = self._quad_drag_coeffs * velocity_body * velocity_body.abs()
        drag_body = linear_drag + quadratic_drag

        proper_force_body = thrust_body - drag_body
        acc_body = proper_force_body / self.m
        acc_world = self._orientation.rotate(acc_body) + self._gravity
        acc_world = acc_world + torch.randn_like(acc_world) * self._acc_noise_std
        self._acc = acc_world

        omega = self._angular_velocity
        inertia = self._inertia
        coriolis = torch.linalg.cross(omega, omega @ inertia.T, dim=1)
        body_torque = self._angular_acc @ inertia.T + coriolis

        def derivatives_fn(*, pos, ori, vel, ori_vel):
            return self.get_derivatives(
                pos=pos,
                ori=ori,
                vel=vel,
                ori_vel=ori_vel,
                acc=acc_world,
                tau=body_torque,
            )

        (
            self._position,
            self._orientation,
            self._velocity,
            self._angular_velocity,
            _,
        ) = Integrator.integrate(
            pos=self._position,
            ori=self._orientation,
            vel=self._velocity,
            ori_vel=self._angular_velocity,
            dt=dt,
            type=self._integrator,
            get_derivatives=derivatives_fn,
        )

    def _ugly_fix(self) -> None:
        self._position[:, :2] = self._position[:, :2].clamp(
            min=self._bd_pos.min, max=self._bd_pos.max
        )
        self._position[:, 2] = self._position[:, 2].clamp(min=0.0, max=self._bd_pos.max)
        self._velocity = self._velocity.clamp(
            min=self._bd_spd.min, max=self._bd_spd.max
        )
        self._angular_velocity = self._angular_velocity.clamp(
            min=self._bd_rate.min, max=self._bd_rate.max
        )
        self._thrust_state = self._thrust_state.clamp(min=0.0, max=self._max_total_thrust)

    # --------------------------------------------------------------------- #
    # Utility functions
    # --------------------------------------------------------------------- #
    def get_derivatives(
        self,
        pos: torch.Tensor,
        ori: Quaternion,
        vel: torch.Tensor,
        ori_vel: torch.Tensor,
        *,
        acc: torch.Tensor,
        tau: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        d_pos = vel
        zeros = torch.zeros(ori_vel.shape[0], device=self.device, dtype=self.dtype)
        omega_quat = Quaternion(zeros, ori_vel[:, 0], ori_vel[:, 1], ori_vel[:, 2])
        d_ori = (ori * omega_quat * 0.5).toTensor()
        d_vel = acc
        J_omega = ori_vel @ self._inertia.T
        coriolis = torch.linalg.cross(ori_vel, J_omega, dim=1)
        d_ori_vel = (tau - coriolis) @ self._inertia_inv.T
        return d_pos, d_ori, d_vel, d_ori_vel

    def _compute_rotor_omega(self, thrusts: torch.Tensor) -> torch.Tensor:
        thrusts = thrusts.clamp(min=0.0, max=self._bd_thrust.max)
        a, b, c = self._rotor_speed_to_thrust_coefs
        discriminant = torch.clamp(b * b - 4 * a * (c - thrusts), min=0.0)
        omega = (-b + torch.sqrt(discriminant)) / (2 * a)
        return omega.clamp(min=self._bd_rotor_omega.min, max=self._bd_rotor_omega.max)

    def set_seed(self, seed: int | None = 42) -> None:
        if seed is None:
            return
        torch.manual_seed(int(seed))

    def close(self) -> None:
        return

    def _get_scale_factor(self) -> None:
        thrust_normalizer = Uniform(
            mean=-self._gravity[:, 2], radius=0.9 * -self._gravity[:, 2]
        ).to(self.device)
        bodyrate_normalizer = Uniform.from_min_max(
            self._bd_rate.min, self._bd_rate.max
        ).to(self.device)
        self._normals = {
            "acc_thrust": thrust_normalizer,
            "bodyrate": bodyrate_normalizer,
        }

    def _de_normalize(self, action: torch.Tensor) -> torch.Tensor:
        if self.action_type != ACTION_TYPE.BODYRATE:
            raise ValueError("Only bodyrate control is implemented.")

        return torch.cat(
            [
                self._normals["acc_thrust"].per_sample_denormalize(action[:, :1]),
                self._normals["bodyrate"].per_sample_denormalize(action[:, 1:]),
            ],
            dim=1,
        )

    # --------------------------------------------------------------------- #
    # Properties
    # --------------------------------------------------------------------- #
    @property
    def position(self) -> torch.Tensor:
        return self._position

    @property
    def orientation(self) -> torch.Tensor:
        return self._orientation.toTensor()

    @property
    def quaternion(self) -> Quaternion:
        return self._orientation

    @property
    def direction(self) -> torch.Tensor:
        return self._orientation.x_axis

    @property
    def velocity(self) -> torch.Tensor:
        return self._velocity

    @property
    def angular_velocity(self) -> torch.Tensor:
        return self._angular_velocity

    @property
    def acceleration(self) -> torch.Tensor:
        return self._acc

    @property
    def angular_acceleration(self) -> torch.Tensor:
        return self._angular_acc

    @property
    def angular_jerk(self) -> torch.Tensor:
        return self._angular_jerk

    @property
    def t(self) -> torch.Tensor:
        return self._t

    @property
    def motor_omega(self) -> torch.Tensor:
        return self._motor_omega

    @property
    def thrusts(self) -> torch.Tensor:
        return self._thrusts

    @property
    def thrust_state(self) -> torch.Tensor:
        return self._thrust_state

    @property
    def angular_velocity_state(self) -> torch.Tensor:
        return self._angular_velocity_state

    @property
    def state(self) -> torch.Tensor:
        return torch.cat(
            [self.position, self.orientation, self.velocity, self.angular_velocity],
            dim=1,
        )

    @property
    def full_state(self) -> torch.Tensor:
        return torch.cat(
            [
                self.position,
                self.orientation,
                self.velocity,
                self.angular_velocity,
                self.thrust_state,
                self.angular_velocity_state,
            ],
            dim=1,
        )

    @property
    def R(self) -> torch.Tensor:
        return self._orientation.R

    @property
    def xz_axis(self) -> torch.Tensor:
        return self._orientation.xz_axis


__all__ = ["Dynamics"]
