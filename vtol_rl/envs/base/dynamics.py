import torch

from vtol_rl import config
from vtol_rl.utils.maths import Integrator, Quaternion
from vtol_rl.utils.type import ACTION_TYPE, PID, Uniform, action_type_alias, bound


class Dynamics:
    g = torch.tensor([[0, 0, -9.81]])  # gravity vector
    z = torch.tensor([[0, 0, 1]])  # upward vector

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
        device: torch.device = torch.device("cpu"),
        integrator: str = "euler",
        drag_random: float = 0,
        cfg: str = "drone_state",
    ):
        assert action_type in ["bodyrate"]
        assert ori_output_type in ["quaternion"]

        self.device = device

        # iterative variables
        self.num: int = num
        self._position: torch.Tensor = torch.Tensor(self.num, 3).to(self.device)
        self._orientation: Quaternion = Quaternion(num=self.num, device=self.device)
        self._velocity: torch.Tensor = torch.Tensor(self.num, 3).to(self.device)
        self._angular_velocity: torch.Tensor = torch.Tensor(self.num, 3).to(self.device)
        self._acc: torch.Tensor = torch.Tensor(self.num, 3).to(self.device)
        self._acc_thrust: torch.Tensor = torch.Tensor(self.num, 1).to(self.device)
        self._angular_acc: torch.Tensor = torch.Tensor(self.num, 3).to(self.device)
        self._angular_jerk: torch.Tensor = torch.Tensor(self.num, 3).to(self.device)
        self._t: torch.Tensor = torch.zeros((self.num,), device=self.device)
        self._thrusts: torch.Tensor | None = None
        self._motor_omega: torch.Tensor | None = None

        self.action_type = action_type_alias[action_type]
        self.sim_time_step = sim_time_step
        self.ctrl_period = ctrl_period

        if self.ctrl_period < self.sim_time_step:
            raise ValueError(
                "ctrl_period must be >= sim_time_step; it is recommended to be multiples of sim_time_step"
            )
        if (
            not torch.as_tensor(self.ctrl_period) % torch.as_tensor(self.sim_time_step)
            == 0
        ):
            raise ValueError("ctrl_period should be a multiple of sim_time_step")

        self._interval_steps = int(self.ctrl_period / self.sim_time_step)
        self._comm_delay_steps = int(comm_delay / self.ctrl_period)
        self._integrator = "1st_order_euler" if integrator == "euler" else integrator
        self._ctrl_delay = ctrl_delay

        # initialization
        self.cfg = cfg
        self.set_seed(seed)
        self._init()
        self._get_scale_factor()
        self._set_device(device)

        # Hover thrust per rotor should be scalar; previous implementation returned a
        # length-3 tensor because gravity is stored as a 3-vector.
        self._init_thrust_mag = (self.m * -Dynamics.g[0, 2]) / 4
        self._init_motor_omega = self._compute_rotor_omega(self._init_thrust_mag)
        self._drag_random = drag_random

    def _init(self):
        try:
            self._load()
        except FileNotFoundError as e:
            print(f"Error loading config file for Dynamics: {e.filename}")
            raise

        # 2(CW) ... 3(CCW)
        # 1(CCW) ... 0(CW)
        motor_direction = torch.tensor(
            [
                [1, -1, -1, 1],
                [-1, -1, 1, 1],
                [0, 0, 0, 0.0],
            ]
        )
        motor_direction = motor_direction / motor_direction.norm(dim=0)
        t_BM_ = self._arm_length * motor_direction

        self._inertia = torch.diag(self._inertia)

        self._inertia_inv = torch.inverse(self._inertia)
        self._B_allocation = torch.cat(
            [
                torch.ones(1, 4),
                t_BM_[:2],
                self._kappa * torch.tensor([1, -1, 1, -1]).unsqueeze(0),
            ],
            dim=0,
        )
        self._B_allocation_inv = torch.inverse(self._B_allocation)

        self._pre_action = [
            torch.zeros((self.num, 4), device=self.device)
            for _ in range(self._comm_delay_steps)
        ]

        self._linear_drag_coeffs = self._linear_drag_coeffs_mean
        self._quad_drag_coeffs = self._quad_drag_coeffs_mean

    def detach(self):
        self._position = self._position.clone().detach()
        self._orientation = self._orientation.clone().detach()
        self._velocity = self._velocity.clone().detach()
        self._angular_velocity = self._angular_velocity.clone().detach()
        if self._motor_omega is not None:
            self._motor_omega = self._motor_omega.clone().detach()
        if self._thrusts is not None:
            self._thrusts = self._thrusts.clone().detach()
        self._angular_acc = self._angular_acc.clone().detach()
        self._acc = self._acc.clone().detach()
        self._t = self._t.clone().detach()
        self._pre_action = [pre_act.clone().detach() for pre_act in self._pre_action]

    def _set_device(self, device):
        self._c = self._c.to(device)
        self._rotor_speed_to_thrust_coefs = self._rotor_speed_to_thrust_coefs.to(device)
        self._B_allocation = self._B_allocation.to(device)
        self._B_allocation_inv = self._B_allocation_inv.to(device)
        self.m = self.m.to(device)
        self._inertia = self._inertia.to(device)
        self._inertia_inv = self._inertia_inv.to(device)
        self._quad_drag_coeffs_mean = self._quad_drag_coeffs_mean.to(device)
        self._linear_drag_coeffs_mean = self._linear_drag_coeffs_mean.to(device)
        self._orientation = self._orientation.to(device)

        # Move drag coefficients if they exist
        if hasattr(self, "_linear_drag_coeffs"):
            self._linear_drag_coeffs = self._linear_drag_coeffs.to(device)
        if hasattr(self, "_quad_drag_coeffs"):
            self._quad_drag_coeffs = self._quad_drag_coeffs.to(device)

        # Move PID controllers to the correct device
        self._BODYRATE_PID = self._BODYRATE_PID.to(device)

        Dynamics.z.to(device)
        Dynamics.g.to(device)

    def _reset_batch_size(self, indices) -> int:
        if indices is None:
            return self.num
        if isinstance(indices, torch.Tensor):
            return int(indices.numel())
        return len(indices)

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
            tensor = value.to(device=self.device, dtype=torch.float32)
        else:
            tensor = torch.as_tensor(value, dtype=torch.float32, device=self.device)

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

    def _coerce_reset_times(
        self,
        value,
        batch_size: int,
    ) -> torch.Tensor | None:
        if value is None:
            return None

        if isinstance(value, torch.Tensor):
            tensor = value.to(device=self.device, dtype=torch.float32)
        else:
            tensor = torch.as_tensor(value, dtype=torch.float32, device=self.device)

        if tensor.ndim == 2 and tensor.shape[-1] == 1:
            tensor = tensor.squeeze(-1)

        if tensor.ndim != 1:
            raise ValueError(
                f"t must have shape (batch,); received {tuple(tensor.shape)}"
            )

        if tensor.shape[0] != batch_size:
            raise ValueError(
                f"t batch dimension must match number of indices ({batch_size}); received {tuple(tensor.shape)}"
            )

        return tensor.contiguous()

    def reset(
        self,
        pos: list | torch.Tensor | None = None,
        ori: list | torch.Tensor | None = None,
        vel: list | torch.Tensor | None = None,
        ori_vel: list | torch.Tensor | None = None,
        motor_omega: list | torch.Tensor | None = None,
        thrusts: list | torch.Tensor | None = None,
        t: list | torch.Tensor | None = None,
        indices: list | None = None,
    ):
        batch_size = self._reset_batch_size(indices)
        pos_tensor = self._coerce_reset_tensor("pos", pos, 3, batch_size)
        vel_tensor = self._coerce_reset_tensor("vel", vel, 3, batch_size)
        ori_vel_tensor = self._coerce_reset_tensor("ori_vel", ori_vel, 3, batch_size)
        motor_tensor = self._coerce_reset_tensor(
            "motor_omega", motor_omega, 4, batch_size
        )
        thrust_tensor = self._coerce_reset_tensor("thrusts", thrusts, 4, batch_size)
        ori_tensor = self._coerce_reset_tensor("ori", ori, 4, batch_size)
        time_tensor = self._coerce_reset_times(t, batch_size)

        if indices is None:
            self._position = (
                torch.zeros((self.num, 3), device=self.device)
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
                torch.zeros((self.num, 3), device=self.device)
                if vel_tensor is None
                else vel_tensor
            )
            self._angular_velocity = (
                torch.zeros((self.num, 3), device=self.device)
                if ori_vel_tensor is None
                else ori_vel_tensor
            )
            self._thrusts = (
                torch.ones((self.num, 4), device=self.device) * self._init_thrust_mag
                if thrust_tensor is None
                else thrust_tensor
            )
            self._motor_omega = (
                torch.ones((self.num, 4), device=self.device) * self._init_motor_omega
                if motor_tensor is None
                else motor_tensor
            )
            self._t = (
                torch.zeros((self.num,), device=self.device)
                if time_tensor is None
                else time_tensor
            )

            self._angular_acc = torch.zeros((self.num, 3), device=self.device)
            self._acc = torch.zeros((self.num, 3), device=self.device)
            self._pre_action = [
                torch.zeros((self.num, 4), device=self.device)
                for _ in range(self._comm_delay_steps)
            ]
            if self._drag_random:
                self._linear_drag_coeffs = self._linear_drag_coeffs_mean * (
                    (
                        (torch.rand_like(self._linear_drag_coeffs_mean) - 0.5)
                        * 2
                        * self._drag_random
                    ).clamp(-0.5, 0.5)
                    + 1
                )
                self._quad_drag_coeffs = self._quad_drag_coeffs_mean * (
                    (
                        (torch.rand_like(self._quad_drag_coeffs_mean) - 0.5)
                        * 2
                        * self._drag_random
                    ).clamp(-0.5, 0.5)
                    + 1
                )

        else:
            idx_count = batch_size
            self._position[indices, :] = (
                torch.zeros((idx_count, 3), device=self.device)
                if pos_tensor is None
                else pos_tensor
            )
            self._orientation[indices] = (
                Quaternion(num=idx_count, device=self.device)
                if ori_tensor is None
                else ori_tensor
            )
            self._velocity[indices, :] = (
                torch.zeros((idx_count, 3), device=self.device)
                if vel_tensor is None
                else vel_tensor
            )
            self._angular_velocity[indices, :] = (
                torch.zeros((idx_count, 3), device=self.device)
                if ori_vel_tensor is None
                else ori_vel_tensor
            )
            self._motor_omega[indices, :] = (
                torch.ones((idx_count, 4), device=self.device) * self._init_motor_omega
                if motor_tensor is None
                else motor_tensor
            )
            self._thrusts[indices, :] = (
                torch.ones((idx_count, 4), device=self.device) * self._init_thrust_mag
                if thrust_tensor is None
                else thrust_tensor
            )
            self._t[indices] = (
                torch.zeros((idx_count,), device=self.device)
                if time_tensor is None
                else time_tensor
            )
            self._t[indices] = (
                torch.zeros((idx_count,), device=self.device)
                + torch.rand((idx_count,)) * 3.14 * 2
                if time_tensor is None
                else time_tensor
            )

            self._angular_acc[indices, :] = torch.zeros(
                (idx_count, 3), device=self.device
            )
            self._acc[indices, :] = torch.zeros((idx_count, 3), device=self.device)
            for i in range(self._comm_delay_steps):
                self._pre_action[i][indices] = 0

            # REC MARK: disable now.
            self._drag_random = False
            if self._drag_random:
                self._linear_drag_coeffs[indices, :] = self._linear_drag_coeffs_mean * (
                    (
                        (
                            torch.rand_like(self._linear_drag_coeffs_mean[indices, :])
                            - 0.5
                        )
                        * 2
                        * self._drag_random
                    ).clamp(-0.5, 0.5)
                    + 1
                )
                self._quad_drag_coeffs[:, indices] = self._quad_drag_coeffs_mean * (
                    (
                        (torch.rand_like(self._quad_drag_coeffs_mean[:, indices]) - 0.5)
                        * 2
                        * self._drag_random
                    ).clamp(-0.5, 0.5)
                    + 1
                )

        return self.state

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
        """
        Compute state derivatives for the current dynamics model.

        Args:
            pos: position in world frame, shape (N, 3)
            ori: orientation quaternion, Quaternion
            vel: linear velocity in world frame, shape (N, 3)
            ori_vel: angular velocity in body frame, shape (N, 3)
            acc: linear acceleration in world frame, shape (N, 3)
            tau: body-frame torque, shape (N, 3)
        """
        d_pos = vel
        omega = ori_vel
        zeros = torch.zeros(omega.shape[0], device=omega.device, dtype=omega.dtype)
        omega_quat = Quaternion(zeros, omega[:, 0], omega[:, 1], omega[:, 2])
        d_ori = (ori * omega_quat * 0.5).toTensor()
        d_vel = acc
        J_omega = omega @ self._inertia.T
        coriolis = torch.linalg.cross(omega, J_omega, dim=1)
        d_ori_vel = (tau - coriolis) @ self._inertia_inv.T
        return d_pos, d_ori, d_vel, d_ori_vel

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Step the simulation forward by one control period given the action.
        Args:
            action (torch.Tensor): shape (N, 4), in range [-1, 1]
        Returns:
            state (torch.Tensor): shape (N, 13): (p,q,v,r)
        """

        if self._comm_delay_steps:
            self._pre_action.append(action.clone())
            action = self._pre_action.pop(0)

        command = self._de_normalize(action)  # command is actual [T, bodyrate_x,y,z]

        thrust_des = self._get_thrust_from_cmd(command)  #

        for _ in range(self._interval_steps):
            # thrust_des = self._get_thrust_from_cmd(command)
            # assert (thrust_des <= self._bd_thrust.max).all()
            self._run_motors(thrust_des)
            # force_torque = self._B_allocation @ self._thrusts  # 计算力矩
            force_torque = self._thrusts @ self._B_allocation.T

            # compute linear acceleration and body torque
            velocity_body = self._orientation.inv_rotate(self._velocity)
            linear_drag_coeffs = self._linear_drag_coeffs.reshape(-1, 3)
            quadratic_drag_coeffs = self._quad_drag_coeffs.reshape(-1, 3)
            linear_drag = linear_drag_coeffs * velocity_body
            quadratic_drag = quadratic_drag_coeffs * velocity_body * velocity_body.abs()
            drag = linear_drag + quadratic_drag
            # drag = self._drag_coeffs * (self._orientation.inv_rotate(self._velocity - 0).pow(2))
            thrust_body = Dynamics.z * force_torque[:, 0].unsqueeze(1)
            proper_force_body = thrust_body - drag
            gravity = Dynamics.g
            self._acc = self._orientation.rotate(proper_force_body) / self.m + gravity

            acc_noise = torch.randn_like(self._acc) * 0.5

            torque = force_torque[:, 1:]

            # integrate the state
            acc_total = self._acc + acc_noise
            torque_total = torque

            def derivatives_fn(*, pos, ori, vel, ori_vel):
                return self.get_derivatives(
                    pos=pos,
                    ori=ori,
                    vel=vel,
                    ori_vel=ori_vel,
                    acc=acc_total,
                    tau=torque_total,
                )

            (
                self._position,
                self._orientation,
                self._velocity,
                self._angular_velocity,
                self._angular_acc,
            ) = Integrator.integrate(
                pos=self._position,
                ori=self._orientation,
                vel=self._velocity,
                ori_vel=self._angular_velocity,
                dt=self.sim_time_step,
                type=self._integrator,
                get_derivatives=derivatives_fn,
            )

        self._t += self.ctrl_period

        self._ugly_fix()  # Re-enabled to prevent position explosion
        print(f"self.state: {self.state}")
        return self.state

    def _ugly_fix(self):
        # Clamp to reasonable values for FSC compatibility
        # X, Y: [-100, 100] meters (large enough for any reasonable scenario)
        # Z: [0, 10] meters (ground to reasonable altitude)
        self._position[0:2] = self._position[0:2].clamp(-100, 100)  # X, Y
        self._position[2] = self._position[2].clamp(0, 10)  # Z (altitude)
        self._velocity = self._velocity.clamp(-10, 10)
        self._angular_velocity = self._angular_velocity.clamp(-10, 10)

    def _get_thrust_from_cmd(self, command) -> torch.Tensor:
        """
        Args:
            command: shape (N, 4)
        Returns:
            thrusts_des: shape (N, 4)
        """

        if self.action_type == ACTION_TYPE.BODYRATE:
            angular_velocity_error = command[:, 1:] - self._angular_velocity

            pid_p = self._BODYRATE_PID.p
            p_term = angular_velocity_error @ pid_p.T

            pid_d = self._BODYRATE_PID.d
            d_term = -(self._angular_acc @ pid_d.T)

            p_term_reshaped = p_term.unsqueeze(2)  # Shape becomes (N, 3, 1)
            inertia_reshaped = self._inertia.unsqueeze(0)  # Shape becomes (1, 3, 3)

            # The result of this matmul will be (N, 3, 1), we then squeeze it back to (N, 3)
            body_torque_des = (
                torch.matmul(inertia_reshaped, p_term_reshaped).squeeze(2) - d_term
            )

            gross_thrust_and_torques = torch.cat(
                [command[:, 0].unsqueeze(1), body_torque_des], dim=1
            )

            thrusts_des = gross_thrust_and_torques @ self._B_allocation_inv.T
        else:
            raise ValueError("action_type should be one of ['bodyrate']")

        clamp_thrusts_des = torch.clamp(
            thrusts_des, self._bd_thrust.min, self._bd_thrust.max
        )

        return clamp_thrusts_des

    def _run_motors(self, thrusts_des) -> torch.Tensor:
        """_summary_
        Returns:
            _type_: _description_
        """
        if self._ctrl_delay:
            motor_omega_des = self._compute_rotor_omega(thrusts_des)

            # simulate motors as a first-order system
            self._motor_omega = (
                self._c * self._motor_omega + (1 - self._c) * motor_omega_des
            )

            self._thrusts = self._compute_thrust(self._motor_omega)
        else:
            self._thrusts = thrusts_des

        return self._thrusts

    def _compute_thrust(self, _motor_omega) -> torch.Tensor:
        """_summary_
            compute the thrust from the motor omega
        Args:
            _motor_omega (_type_): _description_
        Returns:
            _type_: _description_
        """
        _thrusts = (
            (self._rotor_speed_to_thrust_coefs[0] * (_motor_omega).pow(2))
            + self._rotor_speed_to_thrust_coefs[1] * _motor_omega
            + self._rotor_speed_to_thrust_coefs[2]
        )
        return _thrusts

    def _compute_rotor_omega(self, _thrusts) -> torch.Tensor:
        """
        Compute rotor angular speed omega from desired thrust(s) by inverting:
            T(omega) = a*omega^2 + b*omega + c
        This is a element-wise operation. Uses the positive root of the quadratic formula.

        Args:
            thrusts: Tensor of desired thrust [N], shape (N,4).

        Returns:
            omega: Tensor of angular speed [rad/s], shape (N,4), clamped to >= 0.
        """
        scale = 1 / (2 * self._rotor_speed_to_thrust_coefs[0])  # 1/(2a)
        # yi yuan er ci han shu
        omega = scale * (
            -self._rotor_speed_to_thrust_coefs[1]
            + torch.sqrt(
                self._rotor_speed_to_thrust_coefs[1].pow(2)
                - 4
                * self._rotor_speed_to_thrust_coefs[0]
                * (self._rotor_speed_to_thrust_coefs[2] - _thrusts)
            )
        )  # positive soluitno of quadratic formula
        return omega

    # def get_derivatives(self, command: torch.Tensor) -> tuple[torch.Tensor]:
    #     """
    #     Compute the state derivatives given the current state and action.
    #     Args:
    #         command (torch.Tensor): shape (N, 4), in range [almost g+std(.5g), bodyrate~Uniform(-_bd_rate, _bd_rate)]
    #     Returns:
    #         state_dot (torch.Tensor): shape (N, 13)
    #         in the following order:
    #             pos_dot (torch.Tensor): shape (N, 3)
    #             ori_dot (torch.Tensor): shape (N, 4)
    #             vel_dot (torch.Tensor): shape (N, 3)
    #             ang_vel_dot (torch.Tensor): shape (N, 3)
    #     """
    #     pos_dot = self._velocity
    #     zeros = torch.zeros((self.num,), device=self.device)
    #     omega_quat = Quaternion(
    #         zeros,
    #         self._angular_velocity[:, 0],
    #         self._angular_velocity[:, 1],
    #         self._angular_velocity[:, 2],
    #     ).toTensor()
    #     ori_dot = (0.5 * self._orientation * omega_quat).toTensor()
    #     acc_T_cmd = command[:, :1]  # thrust/m
    #     bodyrate_cmd = command[:, 1:]  # bodyrate_x,y,z

    #     vel_dot = self._acc + torch.randn_like(self._acc) * 0.5

    def set_seed(self, seed=42):
        torch.manual_seed(seed)

    def close(self):
        pass

    def _load(self):
        data = config.load_json(f"drone/{self.cfg}.json")
        self.m = torch.tensor(data["mass"])
        self._cross_sections = torch.tensor([data["cross_sections"]]).T  # 机体横截面积
        self._quad_drag_coeffs_mean = (
            torch.tensor([data["quad_drag_coeffs"]]).T
            * 0.5
            * 1.225
            * self._cross_sections
        )
        self._linear_drag_coeffs_mean = torch.tensor([data["linear_drag_coeffs"]]).T
        self._inertia = torch.tensor(data["inertia"])
        self.name = data["name"]
        self._BODYRATE_PID = PID(
            p=torch.tensor(data["BODYRAYE_PID"]["p"]),
            i=torch.tensor(data["BODYRAYE_PID"]["i"]),
            d=torch.tensor(data["BODYRAYE_PID"]["d"]),
        )

        self._kappa = torch.tensor(data["kappa"])
        self._arm_length = torch.tensor(data["arm_length"])
        self._rotor_speed_to_thrust_coefs = torch.tensor(
            data["rotor_speed_to_thrust_coefs"]
        )
        self._motor_tau_inv = torch.tensor(1 / data["motor_tau"])
        self._c = torch.exp(-self._motor_tau_inv * self.sim_time_step)
        self._bd_rotor_omega = bound(
            max=data["motor_omega_max"],
            min=data["motor_omega_min"],
        )
        self._bd_thrust = bound(
            max=self._rotor_speed_to_thrust_coefs[0] * self._bd_rotor_omega.max**2
            + self._rotor_speed_to_thrust_coefs[1] * self._bd_rotor_omega.max
            + self._rotor_speed_to_thrust_coefs[2],
            min=0,
        )
        self._bd_rate = bound(
            max=torch.tensor(data["max_rate"]), min=torch.tensor(-data["max_rate"])
        )
        self._bd_yaw_rate = bound(
            max=torch.tensor(data["max_rate"]), min=torch.tensor(-data["max_rate"])
        )
        self._bd_spd = bound(
            max=torch.tensor(data["max_spd"]), min=torch.tensor(-data["max_spd"])
        )
        self._bd_pos = bound(
            max=torch.tensor(data["max_pos"]), min=torch.tensor(-data["max_pos"])
        )

    def _get_scale_factor(self):
        """_summary_
            get the transformation parameters for the command
        Args:
            normal_range (-1, 1).
        """
        thrust_normalize_method = "medium"  # "max_min"
        g_val = -Dynamics.g.view(-1)[2]
        if self.action_type == ACTION_TYPE.BODYRATE:
            if thrust_normalize_method == "medium":
                thrust_normalizer = Uniform(
                    mean=g_val,
                    radius=0.9 * g_val,  # hover thrust is half of weight
                ).to(self.device)
            elif thrust_normalize_method == "max_min":
                thrust_normalizer = Uniform.from_min_max(
                    self._bd_thrust.min / self.m, self._bd_thrust.max / self.m
                ).to(self.device)
            else:
                raise ValueError(
                    "thrust_normalize_method should be one of ['medium', 'max_min']"
                )
            self._normals = {
                "acc_thrust": thrust_normalizer,
                "bodyrate": Uniform.from_min_max(
                    self._bd_rate.min, self._bd_rate.max
                ).to(self.device),
            }

        else:
            raise ValueError(
                "action_type should be one of ['thrust', 'bodyrate', 'velocity']"
            )

    def _de_normalize(self, action: torch.Tensor) -> torch.Tensor:
        """
            de-normalize the command to the real value
        Args:
            command torch.Tensor: shape (N, 4)
                1. with normed values in range [-1, 1]
                2. with the semantic order of [thrust/m, bodyrate_x, bodyrate_y, bodyrate_z]
        Returns:
            command torch.Tensor: shape (N, 4)
                with real values
        """
        # REC MARK: we choose this order [T, bx, by, bz]
        # REC MARK: Assume commands are in shape (N, 4)
        if self.action_type == ACTION_TYPE.BODYRATE:
            command = torch.cat(
                [
                    self._normals["acc_thrust"].per_sample_denormalize(action[:, :1]),
                    self._normals["bodyrate"].per_sample_denormalize(action[:, 1:]),
                ],
                dim=1,
            )
        else:
            raise ValueError("action_type should be one of ['bodyrate']")
        return command

    @property
    def position(self):
        """
        Returns:
            pos: shape (N, 3)
        """
        return self._position

    @property
    def orientation(self):
        """
        Returns:
            ori: shape (N, 4)
        """
        return self._orientation.toTensor()

    @property
    def direction(self):
        """
        Returns:
            dir: shape (N, 3), x-axis of the body frame in world frame or called forward-axis
        """
        return self._orientation.x_axis

    @property
    def velocity(self):
        """
        Returns:
            vel: shape (N, 3)
        """
        return self._velocity

    @property
    def angular_velocity(self):
        """
        Returns:
            angular_velocity: shape (N, 3)
        """
        return self._angular_velocity

    @property
    def acceleration(self):
        """
        Returns:
            acc: shape (N, 3)
        """
        return self._acc

    @property
    def angular_acceleration(self):
        """
        Returns:
            angular_acc: shape (N, 3)
        """
        return self._angular_acc

    @property
    def t(self):
        """
        Returns:
            t: shape (N,)
        """
        return self._t

    @property
    def motor_omega(self):
        if self._motor_omega is None:
            return torch.zeros(
                (self.num, 4), device=self.device, dtype=self._position.dtype
            )
        return self._motor_omega

    @property
    def thrusts(self):
        """
        Returns:
            thrusts: shape (N, 4)
        """
        if self._thrusts is None:
            return torch.zeros(
                (self.num, 4), device=self.device, dtype=self._position.dtype
            )
        return self._thrusts

    @property
    def state(self):
        """
        Returns:
            state: shape (N, 13) (p, q, v, r)
        """
        return torch.cat(
            [self.position, self.orientation, self.velocity, self.angular_velocity],
            dim=1,
        )

    @property
    def full_state(self):
        return torch.cat(
            [
                self.position,
                self.orientation,
                self.velocity,
                self.angular_velocity,
                self.motor_omega,
                self.thrusts,
                self.t.unsqueeze(1),
            ],
            dim=1,
        )

    @property
    def R(self):
        return self._orientation.R

    @property
    def xz_axis(self):
        return self._orientation.xz_axis


if __name__ == "__main__":
    env = Dynamics()
    env.reset()
    for _ in range(100):
        action = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        state = env.step(action)
        print(state)
