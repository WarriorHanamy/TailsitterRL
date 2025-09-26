import torch

from vtol_rl import config
from vtol_rl.utils.maths import Integrator, Quaternion
from vtol_rl.utils.type import ACTION_TYPE, PID, Uniform, action_type_alias, bound


# These will be moved to the correct device in _set_device method
class Dynamics:
    g = torch.tensor([[0, 0, -9.81]]).T
    z = torch.tensor([[0, 0, 1]]).T

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
        # assert action_type in ["bodyrate", "thrust", "velocity", "position"]
        # assert ori_output_type in ["quaternion", "euler"]
        assert action_type in ["bodyrate"]
        assert ori_output_type in ["quaternion"]

        self.device = device

        # iterative variables
        self.num = num
        self._position = None
        self._orientation = None
        self._velocity = None
        self._angular_velocity = None
        self._motor_omega = None
        self._thrusts = None
        self._acc = None
        self._angular_acc = None
        self._t = None

        # parameters
        self._is_quat_output = ori_output_type == "quaternion"
        self.action_type = action_type_alias[action_type]
        self.angular_output_type = ori_output_type
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
        self._integrator = integrator
        self._ctrl_delay = ctrl_delay

        # initialization
        self.cfg = cfg
        self.set_seed(seed)
        self._init()
        self._get_scale_factor()
        self._set_device(device)

        self._init_thrust_mag = -(self.m * Dynamics.g / 4)[-1]
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
        self._B_allocation = torch.vstack(
            [torch.ones(1, 4), t_BM_[:2], self._kappa * torch.tensor([1, -1, 1, -1])]
        )
        self._B_allocation_inv = torch.inverse(self._B_allocation)

        self._position = torch.zeros((3, self.num), device=self.device)
        self._orientation = Quaternion(num=self.num, device=self.device)
        self._velocity = torch.zeros((3, self.num), device=self.device)
        self._angular_velocity = torch.zeros((3, self.num), device=self.device)

        self._t = torch.zeros((self.num,), device=self.device)

        self._angular_acc = torch.zeros((3, self.num), device=self.device)
        self._pre_action = [
            torch.zeros((4, self.num), device=self.device)
            for _ in range(self._comm_delay_steps)
        ]

        self._linear_drag_coeffs = self._linear_drag_coeffs_mean
        self._quad_drag_coeffs = self._quad_drag_coeffs_mean

    def detach(self):
        self._position = self._position.clone().detach()
        self._orientation = self._orientation.clone().detach()
        self._velocity = self._velocity.clone().detach()
        self._angular_velocity = self._angular_velocity.clone().detach()
        self._motor_omega = self._motor_omega.clone().detach()
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

        # Move drag coefficients if they exist
        if hasattr(self, "_linear_drag_coeffs"):
            self._linear_drag_coeffs = self._linear_drag_coeffs.to(device)
        if hasattr(self, "_quad_drag_coeffs"):
            self._quad_drag_coeffs = self._quad_drag_coeffs.to(device)

        # Move PID controllers to the correct device
        self._BODYRATE_PID = self._BODYRATE_PID.to(device)
        self._VELOCITY_PID = self._VELOCITY_PID.to(device)
        self._POSITION_PID = self._POSITION_PID.to(device)

        Dynamics.z.to(device)
        Dynamics.g.to(device)

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
        if indices is None:
            self._position = (
                torch.zeros((3, self.num), device=self.device) if pos is None else pos.T
            )
            self._orientation = (
                Quaternion(num=self.num, device=self.device)
                if ori is None
                else Quaternion(*ori.T)
            )
            self._velocity = (
                torch.zeros((3, self.num), device=self.device) if vel is None else vel.T
            )
            self._angular_velocity = (
                torch.zeros((3, self.num), device=self.device)
                if ori_vel is None
                else ori_vel.T
            )
            self._thrusts = (
                torch.ones((4, self.num), device=self.device) * self._init_thrust_mag
                if thrusts is None
                else thrusts.T
            )
            self._motor_omega = (
                torch.ones((4, self.num), device=self.device) * self._init_motor_omega
                if motor_omega is None
                else motor_omega.T
            )
            self._t = torch.zeros((self.num,), device=self.device) if t is None else t
            # self._t = torch.zeros((self.num,), device=self.device) + torch.rand((self.num,), device=self.device)*3.14*2 if t is None else t
            # self._ctrl_i = torch.zeros((3, self.num), device=self.device)
            self._angular_acc = torch.zeros((3, self.num), device=self.device)
            self._acc = torch.zeros((3, self.num), device=self.device)
            self._pre_action = [
                torch.zeros(4, self.num) for _ in range(self._comm_delay_steps)
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
            self._position[:, indices] = (
                torch.zeros((3, len(indices)), device=self.device)
                if pos is None
                else pos.T
            )
            self._orientation[indices] = (
                Quaternion(num=len(indices), device=self.device)
                if ori is None
                else Quaternion(*ori.T)
            )
            self._velocity[:, indices] = (
                torch.zeros((3, len(indices)), device=self.device)
                if vel is None
                else vel.T
            )
            self._angular_velocity[:, indices] = (
                torch.zeros((3, len(indices)), device=self.device)
                if ori_vel is None
                else ori_vel.T
            )
            self._motor_omega[:, indices] = (
                torch.ones((4, len(indices)), device=self.device)
                * self._init_motor_omega
                if motor_omega is None
                else motor_omega.T
            )
            self._thrusts[:, indices] = (
                torch.ones((4, len(indices)), device=self.device)
                * self._init_thrust_mag
                if thrusts is None
                else thrusts.T
            )
            self._t[indices] = (
                torch.zeros((len(indices),), device=self.device) if t is None else t
            )
            self._t[indices] = (
                torch.zeros((len(indices),), device=self.device)
                + torch.rand((len(indices),)) * 3.14 * 2
                if t is None
                else t
            )
            # self._ctrl_i[:, indices] = torch.zeros((3, len(indices)), device=self.device)
            self._angular_acc[:, indices] = torch.zeros(
                (3, len(indices)), device=self.device
            )
            self._acc[:, indices] = torch.zeros((3, len(indices)), device=self.device)
            for i in range(self._comm_delay_steps):
                self._pre_action[i][:, indices] = self._pre_action[i][:, indices] * 0

            if self._drag_random:
                self._linear_drag_coeffs[:, indices] = self._linear_drag_coeffs_mean * (
                    (
                        (
                            torch.rand_like(self._linear_drag_coeffs_mean[:, indices])
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

    def _normalize(self, action):
        """
        Normalize the action from real values to [-1, 1] range
        This function is used only for ROS node !!!

        Args:
            action: Real action values to be normalized

        Returns:
            Normalized action in [-1, 1] range
        """
        if not isinstance(action, torch.Tensor):
            action = torch.from_numpy(action)

        action = action.clone()

        if self.action_type == ACTION_TYPE.BODYRATE:
            # action format: [thrust/m, bodyrate_x, bodyrate_y, bodyrate_z]
            normalized = torch.hstack(
                [
                    (action[:, :1] / self.m - self._normals["thrust"].mean)
                    / self._normals["thrust"].radius,
                    (action[:, 1:] - self._normals["bodyrate"].mean)
                    / self._normals["bodyrate"].radius,
                ]
            )
            return normalized
        else:
            raise ValueError(f"Unsupported action_type: {self.action_type}")

    def step(self, action) -> tuple[torch.Tensor, torch.Tensor]:
        # Add real imu delay
        if self._comm_delay_steps:
            self._pre_action.append(action.T.clone())
            action = self._pre_action[0].T
            self._pre_action.pop(0)
        else:
            action = action
        # print(f"action: {action.shape} {action}")
        command = self._de_normalize(action.to(self.device))
        # print(f"command: {command.shape} {command}")
        thrust_des = self._get_thrust_from_cmd(command)  #
        assert (thrust_des <= self._bd_thrust.max).all()  # debug

        for _ in range(self._interval_steps):
            # thrust_des = self._get_thrust_from_cmd(command)
            # assert (thrust_des <= self._bd_thrust.max).all()
            self._run_motors(thrust_des)
            force_torque = self._B_allocation @ self._thrusts  # 计算力矩

            # compute linear acceleration and body torque
            velocity_body = self._orientation.inv_rotate(self._velocity)  # (3, N)
            linear_drag = self._linear_drag_coeffs * velocity_body
            quadratic_drag = (
                self._quad_drag_coeffs * velocity_body * velocity_body.abs()
            )
            drag = linear_drag + quadratic_drag
            # drag = self._drag_coeffs * (self._orientation.inv_rotate(self._velocity - 0).pow(2))
            self._acc = (
                self._orientation.rotate(Dynamics.z * force_torque[0] - drag) / self.m
                + Dynamics.g
            )

            torque = force_torque[1:]

            # integrate the state
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
                acc=self._acc,
                tau=torque,
                J=self._inertia,
                J_inv=self._inertia_inv,
                dt=self.sim_time_step,
                type=self._integrator,
            )
            self._orientation = self._orientation.normalize()
        self._t += self.ctrl_period

        self._ugly_fix()  # Re-enabled to prevent position explosion

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
        """_summary_
            get the single _thrusts from the command
        Args:
            command (_type_): _description_

        Returns:
            _type_: _description_
        """
        if self.action_type == ACTION_TYPE.THRUST:
            thrusts_des = command
        # REC MARK: I remove position and velocity control for simplicity
        elif self.action_type == ACTION_TYPE.BODYRATE:
            # REC MARK: to understand the batch operations.
            # assert command.shape == (
            #     4,
            #     1,
            # ), f"command shape should be (4, 1), but got {command.shape}"
            command = command.squeeze(0)  # (4, N)
            angular_velocity_error = command[1:, :] - self._angular_velocity
            # self._ctrl_i += (self._BODYRATE_PID.i @ (angular_velocity_error * self.sim_time_step))
            # self._ctrl_i = self._ctrl_i.clip(min=-3, max=3)
            body_torque_des = (
                self._inertia @ self._BODYRATE_PID.p @ angular_velocity_error
                - self._BODYRATE_PID.d @ self._angular_acc
            )
            # + cross(self._angular_velocity + 0, self._inertia @ (self._angular_velocity + 0)) \   REC MARK: I don't like this term.
            # + self._ctrl_i \

            # REC MARK: renaming for clarity & make concatenation explicit.
            gross_thrust_and_torques = torch.cat(
                [command[0:1, :], body_torque_des], dim=0
            )
            thrusts_des = self._B_allocation_inv @ gross_thrust_and_torques
        else:
            raise ValueError(
                "action_type should be one of ['thrust', 'bodyrate', 'velocity', 'position']"
            )

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
            thrusts: Tensor of desired thrust [N], shape (4,1).

        Returns:
            omega: Tensor of angular speed [rad/s], shape (4,1), clamped to >= 0.
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
        self._VELOCITY_PID = PID(
            p=torch.tensor(data["VELOCITY_PID"]["p"]),
            i=torch.tensor(data["VELOCITY_PID"]["i"]),
            d=torch.tensor(data["VELOCITY_PID"]["d"]),
        )
        self._POSITION_PID = PID(
            p=torch.tensor(data["POSITION_PID"]["p"]),
            i=torch.tensor(data["POSITION_PID"]["i"]),
            d=torch.tensor(data["POSITION_PID"]["d"]),
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
        if self.action_type == ACTION_TYPE.BODYRATE:
            max_bias = 1
            if thrust_normalize_method == "medium":
                thrust_normalizer = Uniform(
                    mean=-max_bias * Dynamics.g[2], radius=-0.5 * Dynamics.g[2]
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
                "thrust": thrust_normalizer,
                "bodyrate": Uniform.from_min_max(
                    self._bd_rate.min, self._bd_rate.max
                ).to(self.device),
            }

        else:
            raise ValueError(
                "action_type should be one of ['thrust', 'bodyrate', 'velocity']"
            )

    def _de_normalize(self, command):
        """_summary_
            de-normalize the command to the real value
        Args:
            command (_type_): _description_

        Returns:
            _type_: _description_
        """
        if not isinstance(command, torch.Tensor):
            return self._de_normalize(torch.from_numpy(command))

        # REC MARK: we choose this order [T, bx, by, bz]
        if self.action_type == ACTION_TYPE.BODYRATE:
            print(f"command before denormalize: {command}")
            print(f"shape of command: {command.shape}")
            print(f"command[:, :1]: {command[:, :1]}")
            command = torch.hstack(
                [
                    # TODO: denormalize
                    (
                        command[:, :1] * self._normals["thrust"].radius
                        + self._normals["thrust"].mean
                    )
                    * self.m,
                    command[:, 1:] * self._normals["bodyrate"].radius
                    + self._normals["bodyrate"].mean,
                ]
            )
            return command.T
        else:
            raise ValueError(
                "action_type should be one of ['thrust', 'bodyrate', 'velocity', 'position']"
            )

    @property
    def position(self):
        return self._position.T

    @property
    def orientation(self):
        if self._is_quat_output:
            return self._orientation.toTensor().T
        else:
            return self._orientation.toEuler().T

    @property
    def direction(self):
        return self._orientation.x_axis.T

    @property
    def velocity(self):
        return self._velocity.T

    @property
    def angular_velocity(self):
        return self._angular_velocity.T

    @property
    def acceleration(self):
        return self._acc.T

    @property
    def angular_acceleration(self):
        return self._angular_acc.T

    @property
    def t(self):
        return self._t

    @property
    def motor_omega(self):
        return self._motor_omega.T

    @property
    def thrusts(self):
        return self._thrusts.T

    @property
    def state(self):
        return torch.hstack(
            [self.position, self.orientation, self.velocity, self.angular_velocity]
        )

    @property
    def is_quat_output(self):
        return self._is_quat_output

    @property
    def full_state(self):
        return torch.hstack(
            [
                self.position,
                self.orientation,
                self.velocity,
                self.angular_velocity,
                self.motor_omega,
                self.thrusts,
                self.t.unsqueeze(1),
            ]
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
