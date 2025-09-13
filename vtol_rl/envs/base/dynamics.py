import numpy as np
from typing import Union, List, Tuple, Optional, Dict
from abc import ABC
import os, io, sys
import json

import torch
from vtol_rl.utils.maths import Quaternion, Integrator, cross
from vtol_rl.utils.type import ACTION_TYPE, action_type_alias, bound, Uniform, PID
from vtol_rl import config


import torch.nn as nn

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
            action_space: Tuple[float, float] = (-1, 1),
            device: torch.device = torch.device("cpu"),
            integrator: str = "euler",
            drag_random: float = 0,
            cfg: str = "drone_state",
    ):
        assert action_type in ["bodyrate", "thrust", "velocity", "position"]  # 对两个变量进行断言检查
        assert ori_output_type in ["quaternion", "euler"]

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
            raise ValueError("ctrl_period must be >= sim_time_step; it is recommended to be multiples of sim_time_step")
        if not torch.as_tensor(self.ctrl_period) % torch.as_tensor(self.sim_time_step) == 0:
            raise ValueError("ctrl_period should be a multiple of sim_time_step")

        self._interval_steps = int(self.ctrl_period / self.sim_time_step)
        self._comm_delay_steps = int(comm_delay / self.ctrl_period)
        self._integrator = integrator
        self._ctrl_delay = ctrl_delay

        # initialization
        self.cfg = cfg
        self.set_seed(seed)
        self._init()
        self._get_scale_factor(action_space)
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
        motor_direction = torch.tensor([
            [1, -1, -1, 1, ],
            [-1, -1, 1, 1],
            [0, 0, 0, 0.],
        ])
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
        self._pre_action = [
            pre_act.clone().detach() for pre_act in self._pre_action
        ]

    def _set_device(self, device):
        self._c = self._c.to(device)
        self._thrust_map = self._thrust_map.to(device)
        self._B_allocation = self._B_allocation.to(device)
        self._B_allocation_inv = self._B_allocation_inv.to(device)
        self.m = self.m.to(device)
        self._inertia = self._inertia.to(device)
        self._inertia_inv = self._inertia_inv.to(device)
        self._quad_drag_coeffs_mean = self._quad_drag_coeffs_mean.to(device)
        self._linear_drag_coeffs_mean = self._linear_drag_coeffs_mean.to(device)
        
        # Move drag coefficients if they exist
        if hasattr(self, '_linear_drag_coeffs'):
            self._linear_drag_coeffs = self._linear_drag_coeffs.to(device)
        if hasattr(self, '_quad_drag_coeffs'):
            self._quad_drag_coeffs = self._quad_drag_coeffs.to(device)
        
        # Move PID controllers to the correct device
        self._BODYRATE_PID = self._BODYRATE_PID.to(device)
        self._VELOCITY_PID = self._VELOCITY_PID.to(device)
        self._POSITION_PID = self._POSITION_PID.to(device)

        Dynamics.z.to(device)
        Dynamics.g.to(device)

    def reset(
            self,
            pos: Union[List, torch.Tensor, None] = None,
            ori: Union[List, torch.Tensor, None] = None,
            vel: Union[List, torch.Tensor, None] = None,
            ori_vel: Union[List, torch.Tensor, None] = None,
            motor_omega: Union[List, torch.Tensor, None] = None,
            thrusts: Union[List, torch.Tensor, None] = None,
            t: Union[List, torch.Tensor, None] = None,
            indices: Optional[List] = None,
    ):
        if indices is None:
            self._position = torch.zeros((3, self.num), device=self.device) if pos is None else pos.T
            self._orientation = Quaternion(num=self.num, device=self.device) if ori is None else Quaternion(*ori.T)
            self._velocity = torch.zeros((3, self.num), device=self.device) if vel is None else vel.T
            self._angular_velocity = torch.zeros((3, self.num), device=self.device) if ori_vel is None else ori_vel.T
            self._thrusts = torch.ones((4, self.num), device=self.device) * self._init_thrust_mag if thrusts is None else thrusts.T
            self._motor_omega = torch.ones((4, self.num), device=self.device) * self._init_motor_omega if motor_omega is None else motor_omega.T
            self._t = torch.zeros((self.num,), device=self.device) if t is None else t
            # self._t = torch.zeros((self.num,), device=self.device) + torch.rand((self.num,), device=self.device)*3.14*2 if t is None else t
            # self._ctrl_i = torch.zeros((3, self.num), device=self.device)
            self._angular_acc = torch.zeros((3, self.num), device=self.device)
            self._acc = torch.zeros((3, self.num), device=self.device)
            self._pre_action = [torch.zeros(4, self.num) for _ in range(self._comm_delay_steps)]
            if self._drag_random:
                self._linear_drag_coeffs = self._linear_drag_coeffs_mean * (((torch.rand_like(self._linear_drag_coeffs_mean)-0.5)*2*self._drag_random).clamp(-0.5, .5) + 1)
                self._quad_drag_coeffs = self._quad_drag_coeffs_mean * (((torch.rand_like(self._quad_drag_coeffs_mean)-0.5)*2*self._drag_random).clamp(-0.5, .5) + 1)

        else:
            self._position[:, indices] = torch.zeros((3, len(indices)), device=self.device) if pos is None else pos.T
            self._orientation[indices] = Quaternion(num=len(indices), device=self.device) if ori is None else Quaternion(*ori.T)
            self._velocity[:, indices] = torch.zeros((3, len(indices)), device=self.device) if vel is None else vel.T
            self._angular_velocity[:, indices] = torch.zeros((3, len(indices)), device=self.device) if ori_vel is None else ori_vel.T
            self._motor_omega[:, indices] = torch.ones((4, len(indices)), device=self.device) * self._init_motor_omega if motor_omega is None else motor_omega.T
            self._thrusts[:, indices] = torch.ones((4, len(indices)), device=self.device) * self._init_thrust_mag if thrusts is None else thrusts.T
            self._t[indices] = torch.zeros((len(indices),), device=self.device) if t is None else t
            self._t[indices] = torch.zeros((len(indices),), device=self.device) + torch.rand((len(indices),)) * 3.14*2 if t is None else t
            # self._ctrl_i[:, indices] = torch.zeros((3, len(indices)), device=self.device)
            self._angular_acc[:, indices] = torch.zeros((3, len(indices)), device=self.device)
            self._acc[:, indices] = torch.zeros((3, len(indices)), device=self.device)
            for i in range(self._comm_delay_steps):
                self._pre_action[i][:, indices] = self._pre_action[i][:, indices] * 0
            
            if self._drag_random:
                self._linear_drag_coeffs[:, indices] = self._linear_drag_coeffs_mean * (((torch.rand_like(self._linear_drag_coeffs_mean[:,indices])-0.5)*2*self._drag_random).clamp(-0.5, .5) + 1)
                self._quad_drag_coeffs[:, indices] = self._quad_drag_coeffs_mean * (((torch.rand_like(self._quad_drag_coeffs_mean[:,indices])-0.5)*2*self._drag_random).clamp(-0.5, .5) + 1)

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
            normalized = torch.hstack([
                (action[:, :1] / self.m - self._normal_params["thrust"].mean) / self._normal_params["thrust"].half,
                (action[:, 1:] - self._normal_params["bodyrate"].mean) / self._normal_params["bodyrate"].half
            ])
            return normalized
            
        elif self.action_type == ACTION_TYPE.THRUST:
            # action format: [thrust]
            normalized = (action / self.m - self._normal_params["thrust"].mean) / self._normal_params["thrust"].half
            return normalized
            
        elif self.action_type == ACTION_TYPE.VELOCITY:
            # action format: [yaw, velocity_x, velocity_y, velocity_z]
            normalized = torch.hstack([
                (action[:, :1] - self._normal_params["yaw"].mean) / self._normal_params["yaw"].half,
                (action[:, 1:] - self._normal_params["velocity"].mean) / self._normal_params["velocity"].half
            ])
            return normalized
            
        elif self.action_type == ACTION_TYPE.POSITION:
            # action format: [yaw, position_x, position_y, position_z]
            normalized = torch.hstack([
                (action[:, :1] - self._normal_params["yaw"].mean) / self._normal_params["yaw"].half,
                (action[:, 1:] - self._normal_params["velocity"].mean) / self._normal_params["velocity"].half
            ])
            return normalized
            
        else:
            raise ValueError(f"Unsupported action_type: {self.action_type}")

    def step(self, action) -> Tuple[torch.Tensor, torch.Tensor]:

        # Add real imu delay
        if self._comm_delay_steps:
            self._pre_action.append(action.T.clone())
            action = self._pre_action[0].T
            self._pre_action.pop(0)
        else:
            action = action

        command = self._de_normalize(action.to(self.device))

        thrust_des = self._get_thrust_from_cmd(command)  #
        assert (thrust_des <= self._bd_thrust.max).all()  # debug

        for _ in range(self._interval_steps):
            # thrust_des = self._get_thrust_from_cmd(command)
            # assert (thrust_des <= self._bd_thrust.max).all()
            self._run_motors(thrust_des)
            force_torque = self._B_allocation @ self._thrusts  # 计算力矩

            # compute linear acceleration and body torque
            velocity_body = self._orientation.inv_rotate(self._velocity+0)  # (3, N)
            linear_drag = self._linear_drag_coeffs * velocity_body
            quadratic_drag = self._quad_drag_coeffs * velocity_body * velocity_body.abs()
            drag = linear_drag + quadratic_drag
            # drag = self._drag_coeffs * (self._orientation.inv_rotate(self._velocity - 0).pow(2))
            self._acc = self._orientation.rotate(Dynamics.z * force_torque[0] - drag) / self.m + Dynamics.g

            torque = force_torque[1:]

            # integrate the state
            self._position, self._orientation, self._velocity, self._angular_velocity, self._angular_acc = (
                Integrator.integrate(
                    pos=self._position,
                    ori=self._orientation,
                    vel=self._velocity,
                    ori_vel=self._angular_velocity,
                    acc=self._acc,
                    tau=torque,
                    J=self._inertia,
                    J_inv=self._inertia_inv,
                    dt=self.sim_time_step,
                    type=self._integrator
                )
            )
            self._orientation = self._orientation.normalize()
        print(self._t)
        print(self.ctrl_period)
        exit()
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
        elif self.action_type == ACTION_TYPE.BODYRATE:
            angular_velocity_error = command[1:] - self._angular_velocity
            # self._ctrl_i += (self._BODYRATE_PID.i @ (angular_velocity_error * self.sim_time_step))
            # self._ctrl_i = self._ctrl_i.clip(min=-3, max=3)
            body_torque_des = \
                self._inertia @ self._BODYRATE_PID.p @ angular_velocity_error \
                + cross(self._angular_velocity + 0, self._inertia @ (self._angular_velocity + 0)) \
                - self._BODYRATE_PID.d @ self._angular_acc
            # + self._ctrl_i \

            thrusts_torque = torch.cat([command[0:1, :], body_torque_des])
            thrusts_des = self._B_allocation_inv @ thrusts_torque
        elif self.action_type == ACTION_TYPE.VELOCITY:
            command = command.T
            a_des = self._VELOCITY_PID.p * (command[1:] - self._velocity)
            F_des = self.m * (a_des - g)  # world axis

            # Auto yaw control - make drone face velocity direction
            velocity_horizontal = self._velocity[:2, :]  # Get x,y components
            velocity_norm = velocity_horizontal.norm(dim=0)
            # Only update yaw when there's significant horizontal movement
            yaw_des = torch.where(
                velocity_norm > 0.1,  # threshold to avoid jitter when stationary
                torch.atan2(velocity_horizontal[1], velocity_horizontal[0]),
                self._orientation.toEuler()[2]  # keep current yaw when stationary
            )

            current_yaw = self._orientation.toEuler()[2]
            yaw_error = yaw_des - current_yaw
            # Handle yaw angle wrapping (keep error in [-π, π])
            yaw_error = torch.atan2(torch.sin(yaw_error), torch.cos(yaw_error))
            yaw_spd_des = yaw_error * self._VELOCITY_PID.d * 2.0  # Gain for yaw tracking

            gross_thrust_des = self._orientation.transform(F_des)[2]
            R = self._orientation.R
            b3_des = F_des / F_des.norm(dim=0)
            c1_des = torch.stack([yaw_des.cos(), yaw_des.sin(), torch.zeros_like(yaw_des)], dim=0)
            b2_des = cross(b3_des, c1_des)
            b2_des = b2_des / b2_des.norm(dim=0)
            b1_des = cross(b2_des, b3_des)
            R_des = torch.stack([b1_des, b2_des, b3_des]).transpose(0, 1)

            pose_err = torch.zeros_like(self._position)
            ang_vel_err = torch.zeros_like(self._position)
            for i in range(self.num):
                m = 0.5 * (R_des[..., i].T @ R[..., i] - R[..., i].T @ R_des[..., i])
                pose_err[:, i] = -torch.as_tensor([-m[1, 2], m[0, 2], -m[0, 1]], device=self.device)

                ang_vel_err[:, i] = (R_des[..., i].T @ R[..., i] @ torch.tensor([[0], [0], [yaw_spd_des[i]]], device=self.device).squeeze() - self._angular_velocity[:, i])
            body_torque_des = self._inertia @ (self._BODYRATE_PID.p @ pose_err + self._BODYRATE_PID.p @ ang_vel_err - cross(self._angular_velocity, self._angular_velocity))

            thrusts_des = self._B_allocation_inv @ torch.vstack([gross_thrust_des, body_torque_des])
        elif self.action_type == ACTION_TYPE.POSITION:
            command = command.T
            v_des = self._POSITION_PID.d * (command[1:] - self._position)
            a_des = self._VELOCITY_PID.d * (v_des - self._velocity)
            F_des = self.m * (a_des - g)  # world axis

            # Use command[0] as desired yaw angle instead of auto yaw control
            yaw_des = command[0]  # Direct yaw control from command input

            current_yaw = self._orientation.toEuler()[2]
            yaw_error = yaw_des - current_yaw
            # Handle yaw angle wrapping (keep error in [-π, π])
            # 确保角度误差在最短路径上，处理角度环绕问题
            yaw_error = torch.atan2(torch.sin(yaw_error), torch.cos(yaw_error))
            yaw_spd_des = yaw_error * self._POSITION_PID.d * 2.0

            gross_thrust_des = self._orientation.transform(F_des)[2]
            R = self._orientation.R
            b3_des = F_des / F_des.norm(dim=0)
            c1_des = torch.stack([yaw_des.cos(), yaw_des.sin(), torch.zeros_like(yaw_des)], dim=0)
            b2_des = cross(b3_des, c1_des)
            b2_des = b2_des / b2_des.norm(dim=0)
            b1_des = cross(b2_des, b3_des)
            R_des = torch.stack([b1_des, b2_des, b3_des]).transpose(0, 1)

            pose_err = torch.zeros_like(self._position)
            ang_vel_err = torch.zeros_like(self._position)
            for i in range(self.num):
                m = 0.5 * (R_des[..., i].T @ R[..., i] - R[..., i].T @ R_des[..., i])
                pose_err[:, i] = -torch.as_tensor([-m[1, 2], m[0, 2], -m[0, 1]], device=self.device)

                ang_vel_err[:, i] = (
                                        R_des[..., i].T @ R[..., i] @ torch.tensor([[0], [0], [yaw_spd_des[i]]], device=self.device).squeeze()
                                        - self._angular_velocity[:, i]
                                     )
            body_torque_des = self._inertia @ (
                    self._BODYRATE_PID.p @ pose_err
                    + 1.2 * self._BODYRATE_PID.p @ ang_vel_err
                    - self._BODYRATE_PID.d @ self._angular_acc
                    - cross(self._angular_velocity, self._inertia @ self._angular_velocity)
            )

            thrusts_des = self._B_allocation_inv @ torch.vstack([gross_thrust_des, body_torque_des])
            # raise NotImplementedError
        else:
            raise ValueError("action_type should be one of ['thrust', 'bodyrate', 'velocity', 'position']")

        clamp_thrusts_des = torch.clamp(thrusts_des, self._bd_thrust.min, self._bd_thrust.max)

        return clamp_thrusts_des

    def _run_motors(self, thrusts_des) -> torch.Tensor:
        """_summary_
        Returns:
            _type_: _description_
        """
        if self._ctrl_delay:
            motor_omega_des = self._compute_rotor_omega(thrusts_des)

            # simulate motors as a first-order system
            self._motor_omega = self._c * self._motor_omega + (1 - self._c) * motor_omega_des

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
                (self._thrust_map[0] * (_motor_omega+0).pow(2))
                + self._thrust_map[1] * _motor_omega
                + self._thrust_map[2]
        )
        return _thrusts

    def _compute_rotor_omega(self, _thrusts) -> torch.Tensor:
        """_summary_
            compute the rotor omega from the _thrusts by solving quadratic equation
        Args:
            thrusts_des (_type_): _description_
        Returns:
            _type_: _description_
        """
        scale = 1 / (2 * self._thrust_map[0])
        # yi yuan er ci han shu
        omega = scale * (
                -self._thrust_map[1]
                + torch.sqrt(
            self._thrust_map[1].pow(2)
            - 4 * self._thrust_map[0] * (self._thrust_map[2] - _thrusts)
        )
        )
        return omega

    def set_seed(self, seed=42):
        torch.manual_seed(seed)

    def close(self):
        pass

    def _load(self):
        data = config.load_json(f"drone/{self.cfg}.json")
        self.m = torch.tensor(data["mass"])
        self._cross_sections = torch.tensor([data["cross_sections"]]).T  # 机体横截面积
        self._quad_drag_coeffs_mean = torch.tensor([data["quad_drag_coeffs"]]).T * 0.5 * 1.225 * self._cross_sections
        self._linear_drag_coeffs_mean = torch.tensor([data["linear_drag_coeffs"]]).T
        self._inertia = torch.tensor(data["inertia"])
        self.name = data["name"]
        self._BODYRATE_PID = PID(p=torch.tensor(data["BODYRAYE_PID"]["p"]),
                                 i=torch.tensor(data["BODYRAYE_PID"]["i"]),
                                 d=torch.tensor(data["BODYRAYE_PID"]["d"]))
        self._VELOCITY_PID = PID(p=torch.tensor(data["VELOCITY_PID"]["p"]), i=torch.tensor(data["VELOCITY_PID"]["i"]), d=torch.tensor(data["VELOCITY_PID"]["d"]))
        self._POSITION_PID = PID(p=torch.tensor(data["POSITION_PID"]["p"]), i=torch.tensor(data["POSITION_PID"]["i"]), d=torch.tensor(data["POSITION_PID"]["d"]))
        self._kappa = torch.tensor(data["kappa"])
        self._arm_length = torch.tensor(data["arm_length"])
        self._thrust_map = torch.tensor(data["thrust_map"])
        self._motor_tau_inv = torch.tensor(1 / data["motor_tau"])
        self._c = torch.exp(-self._motor_tau_inv * self.sim_time_step)
        self._bd_rotor_omega = bound(
            max=data["motor_omega_max"],
            min=data["motor_omega_min"],
        )
        self._bd_thrust = bound(
            max= \
                self._thrust_map[0] * self._bd_rotor_omega.max ** 2
                + self._thrust_map[1] * self._bd_rotor_omega.max
                + self._thrust_map[2]
            ,
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

    def _get_scale_factor(self, normal_range: Tuple[float, float] = (-1, 1)):
        """_summary_
            get the transformation parameters for the command
        Args:
            normal_range (Tuple[float, float], optional): _description_. Defaults to (-1, 1).
        """
        thrust_normalize_method = "medium"  # "max_min"

        if self.action_type == ACTION_TYPE.BODYRATE:
            max_bias = 1
            if thrust_normalize_method == "medium":
                # (_, average_)
                thrust_scale = (self.m * -Dynamics.g[2]) / self.m
                # thrust_scale = (self.m * -g[2]) * 1 / self.m
                thrust_bias = (self.m * -Dynamics.g[2]) * max_bias / self.m
            elif thrust_normalize_method == "max_min":
                # (min_act, max_act)->(min_thrust, max_thrust) this method try to reach the limit of drone, which is negative for sim2real
                thrust_scale = (
                        (self._bd_thrust.max - self._bd_thrust.min)
                        / self.m
                        / (normal_range[1] - normal_range[0])
                )
                thrust_bias = self._bd_thrust.max / self.m - thrust_scale * normal_range[1]
            else:
                raise ValueError("thrust_normalize_method should be one of ['medium', 'max_min']")

            bodyrate_scale = (self._bd_rate.max - self._bd_rate.min) / (
                    normal_range[1] - normal_range[0]
            )
            bodyrate_bias = self._bd_rate.max - bodyrate_scale * normal_range[1]
            self._normal_params = {
                "thrust": Uniform(mean=thrust_bias, half=thrust_scale).to(self.device),
                "bodyrate": Uniform(mean=bodyrate_bias, half=bodyrate_scale).to(self.device),
            }

        elif self.action_type == ACTION_TYPE.THRUST:
            if thrust_normalize_method == "medium":
                # (_, average_)
                scale = (self.m * -g[2]) / 4 * 2 / self.m
                bias = (self.m * -g[2]) / 4 / self.m
            elif thrust_normalize_method == "max_min":
                scale = (
                        (self._bd_thrust.max - self._bd_thrust.min)
                        / self.m
                        / (normal_range[1] - normal_range[0])
                )
                bias = self._bd_thrust.max / self.m - scale * normal_range[1]
            else:
                raise ValueError("thrust_normalize_method should be one of ['medium', 'max_min']")

            self._normal_params = {"thrust": Uniform(mean=bias, half=scale).to(self.device)}

        elif self.action_type == ACTION_TYPE.VELOCITY:
            spd_scale = (self._bd_spd.max - self._bd_spd.min) / (
                    normal_range[1] - normal_range[0]
            )
            spd_bias = self._bd_spd.max - spd_scale * normal_range[1]
            yaw_scale = torch.as_tensor(torch.pi - (-torch.pi)) / (
                    normal_range[1] - normal_range[0]
            )
            yaw_bias = torch.pi - yaw_scale * normal_range[1]
            self._normal_params = {
                "velocity": Uniform(mean=spd_bias, half=spd_scale).to(self.device),
                "yaw": Uniform(mean=yaw_bias, half=yaw_bias).to(self.device),
            }

        elif self.action_type.POSITION:
            pos_scale = (self._bd_pos.max - self._bd_pos.min) / (
                    normal_range[1] - normal_range[0]
            )
            pos_bias = self._bd_pos.max - pos_scale * normal_range[1]
            yaw_scale = torch.as_tensor(torch.pi - (-torch.pi)) / (
                    normal_range[1] - normal_range[0]
            )
            yaw_bias = torch.pi - yaw_scale * normal_range[1]
            self._normal_params = {
                "velocity": Uniform(mean=pos_bias, half=pos_scale).to(self.device),
                "yaw": Uniform(mean=yaw_bias, half=yaw_scale).to(self.device),
            }

        else:
            raise ValueError("action_type should be one of ['thrust', 'bodyrate', 'velocity']")


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

        if self.action_type == ACTION_TYPE.BODYRATE:
            command = torch.hstack([
                (command[:, :1] * self._normal_params["thrust"].half + self._normal_params["thrust"].mean) * self.m,
                command[:, 1:] * self._normal_params["bodyrate"].half + self._normal_params["bodyrate"].mean
            ]
            )
            return command.T

        elif self.action_type == ACTION_TYPE.THRUST:
            command = self.m * (command * self._normal_params["thrust"].half + self._normal_params["thrust"].mean).T
            return command

        elif self.action_type == ACTION_TYPE.VELOCITY:
            command = torch.hstack([
                command[:, :1] * self._normal_params["yaw"].half + self._normal_params["yaw"].mean,
                command[:, 1:] * self._normal_params["velocity"].half + self._normal_params["velocity"].mean
            ]
            )
            return command

        elif self.action_type == ACTION_TYPE.POSITION:
            # Now position commands contain [yaw, x, y, z] with direct yaw control
            command = torch.hstack([
                command[:, :1] * self._normal_params["yaw"].half + self._normal_params["yaw"].mean,  # yaw angle
                command[:, 1:] * self._normal_params["velocity"].half + self._normal_params["velocity"].mean  # position x,y,z
            ])
            return command

        else:
            raise ValueError("action_type should be one of ['thrust', 'bodyrate', 'velocity', 'position']")

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
        return torch.hstack([
            self.position,
            self.orientation,
            self.velocity,
            self.angular_velocity
        ]
        )

    @property
    def is_quat_output(self):
        return self._is_quat_output

    @property
    def full_state(self):
        return torch.hstack([
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
    env = Dynamics(
    )
    env.reset()
    for _ in range(10):
        action = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        state = env.step(action)
        print(env.acceleration)