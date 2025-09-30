import torch


class Quaternion:
    def __init__(
        self,
        w: torch.Tensor | None = None,
        x: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
        num=1,
        device=torch.device("cpu"),
    ):
        _types = (type(w), type(x), type(y), type(z))
        assert all(
            t is _types[0] for t in _types
        ), "w, x, y, z should have the same type"
        if w is None:
            self.w = torch.ones(num, device=device)
            self.x = torch.zeros(num, device=device)
            self.y = torch.zeros(num, device=device)
            self.z = torch.zeros(num, device=device)
        elif isinstance(w, (int, float)):
            self.w = torch.ones(num, device=device) * w
            self.x = torch.ones(num, device=device) * x
            self.y = torch.ones(num, device=device) * y
            self.z = torch.ones(num, device=device) * z
        elif isinstance(w, torch.Tensor):
            self.w = w
            self.x = x
            self.y = y
            self.z = z
        else:
            raise ValueError("unsupported type")

    @staticmethod
    def _split_tensor_components(tensor: torch.Tensor):
        if tensor.ndim == 1:
            if tensor.shape[0] != 4:
                raise ValueError(
                    "Tensor must have 4 elements to represent a quaternion"
                )
            return tensor[0], tensor[1], tensor[2], tensor[3]

        if tensor.ndim == 2:
            if tensor.shape[0] == 4 and tensor.shape[1] != 4:
                return tensor[0], tensor[1], tensor[2], tensor[3]
            if tensor.shape[1] == 4:
                return tensor[:, 0], tensor[:, 1], tensor[:, 2], tensor[:, 3]

        raise ValueError(
            "Tensor must have shape (4,), (4, N), or (N, 4) to represent quaternion components"
        )

    def to(self, device):
        self.w = self.w.to(device)
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        self.z = self.z.to(device)
        return self

    def rotate(self, other):
        if isinstance(other, Quaternion):
            # quaternion multiplication
            return self * other

        if not torch.is_tensor(other):
            other = torch.as_tensor(other, dtype=self.w.dtype, device=self.w.device)

        squeeze_output = False
        transpose_output = False

        if other.ndim == 1:
            if other.shape[0] != 3:
                raise ValueError("Vector must have 3 elements")
            other = other.unsqueeze(0)
            squeeze_output = True
        elif other.ndim == 2:
            if other.shape[1] == 3:
                other = other
            elif other.shape[0] == 3:
                other = other.T
                transpose_output = True
            else:
                raise ValueError("Vector batch must have shape (N, 3) or (3, N)")
        else:
            raise ValueError("Vector must be 1D or 2D tensor")

        other = other.to(device=self.w.device, dtype=self.w.dtype)

        zeros = torch.zeros(other.size(0), device=other.device, dtype=other.dtype)
        pure = Quaternion(zeros, other[:, 0], other[:, 1], other[:, 2])
        rotated = self * pure * self.conjugate()
        result = rotated.imag

        if squeeze_output:
            return result.squeeze(0)
        if transpose_output:
            return result.T
        return result

    def inv_rotate(self, other):
        """
        rotate to local
        """
        if isinstance(other, Quaternion):
            return self.conjugate() * other

        if not torch.is_tensor(other):
            other = torch.as_tensor(other, dtype=self.w.dtype, device=self.w.device)

        squeeze_output = False
        transpose_output = False

        if other.ndim == 1:
            if other.shape[0] != 3:
                raise ValueError("Vector must have 3 elements")
            other = other.unsqueeze(0)
            squeeze_output = True
        elif other.ndim == 2:
            if other.shape[1] == 3:
                pass
            elif other.shape[0] == 3:
                other = other.T
                transpose_output = True
            else:
                raise ValueError("Vector batch must have shape (N, 3) or (3, N)")
        else:
            raise ValueError("Vector must be 1D or 2D tensor")

        other = other.to(device=self.w.device, dtype=self.w.dtype)

        zeros = torch.zeros(other.size(0), device=other.device, dtype=other.dtype)
        pure = Quaternion(zeros, other[:, 0], other[:, 1], other[:, 2])
        rotated = self.conjugate() * pure * self
        result = rotated.imag

        if squeeze_output:
            return result.squeeze(0)
        if transpose_output:
            return result.T
        return result

    def extract_yaw_only(self):
        """
        提取当前四元数中只保留 yaw（绕 Z）的部分，返回一个新的 Quaternion 实例。
        """
        # yaw = atan2(2(wz + xy), 1 - 2(y² + z²))
        yaw = torch.atan2(
            2 * (self.w * self.z + self.x * self.y), 1 - 2 * (self.y**2 + self.z**2)
        )
        half_yaw = yaw / 2

        w = torch.cos(half_yaw)
        z = torch.sin(half_yaw)
        x = torch.zeros_like(w)
        y = torch.zeros_like(w)
        return Quaternion(w, x, y, z)

    def extract_pitch_roll(self):
        """
        提取当前四元数中只保留 pitch 和 roll 的部分，返回一个新的 Quaternion 实例。
        """
        # pitch = atan2(2(wx + yz), 1 - 2(x² + z²))
        # roll = atan2(2(wy - xz), 1 - 2(y² + z²))
        pitch = torch.atan2(
            2 * (self.w * self.y + self.x * self.z), 1 - 2 * (self.x**2 + self.z**2)
        )
        roll = torch.atan2(
            2 * (self.w * self.x - self.y * self.z), 1 - 2 * (self.y**2 + self.z**2)
        )

        half_pitch = pitch / 2
        half_roll = roll / 2

        w = torch.cos(half_pitch) * torch.cos(half_roll)
        x = torch.sin(half_roll) * torch.cos(half_pitch)
        y = torch.sin(half_pitch) * torch.cos(half_roll)
        z = torch.sin(half_pitch) * torch.sin(half_roll)

        return Quaternion(w, x, y, z)

    def world_to_head(self, vec: torch.Tensor) -> torch.Tensor:
        """
        将世界坐标系中的向量 vec 投影到当前四元数定义的航迹坐标系下（仅考虑 yaw）
        """
        q_yaw = self.extract_yaw_only()
        return (
            q_yaw.conjugate()
            * Quaternion(torch.tensor(0.0, device=vec.device), *vec)
            * q_yaw
        ).imag

    def local_to_head(self, vec: torch.Tensor) -> torch.Tensor:
        """
        将局部坐标系中的向量 vec 映射到航迹坐标系下：
        先从 local → world，再从 world → heading
        """
        v_world = self.rotate(vec)
        q_yaw = self.extract_yaw_only()
        return (
            q_yaw.conjugate()
            * Quaternion(torch.tensor(0, device=vec.device), *v_world)
            * q_yaw
        ).imag

    def transform(self, other):
        return self.inv_rotate(other)

    def inv_transform(self, other):
        return self.rotate(other)

    @property
    def R(self):
        """
        Args: self (Quaternion): shape (N, (w,x,y,z))
        Returns: R (torch.Tensor): shape (N, 3, 3)
        """
        r00 = 1 - 2 * (self.y.pow(2) + self.z.pow(2))
        r01 = 2 * (self.x * self.y - self.z * self.w)
        r02 = 2 * (self.x * self.z + self.y * self.w)

        r10 = 2 * (self.x * self.y + self.z * self.w)
        r11 = 1 - 2 * (self.x.pow(2) + self.z.pow(2))
        r12 = 2 * (self.y * self.z - self.x * self.w)

        r20 = 2 * (self.x * self.z - self.y * self.w)
        r21 = 2 * (self.y * self.z + self.x * self.w)
        r22 = 1 - 2 * (self.x.pow(2) + self.y.pow(2))

        row0 = torch.stack([r00, r01, r02], dim=-1)
        row1 = torch.stack([r10, r11, r12], dim=-1)
        row2 = torch.stack([r20, r21, r22], dim=-1)
        return torch.stack([row0, row1, row2], dim=-2)

    @property
    def x_axis(self):
        # return torch.stack([1 - 2 * (self.y.pow(2) + self.z.pow(2)), 2 * (self.x * self.y + self.z * self.w), 2 * (self.x * self.z - self.y * self.w)])
        return torch.stack(
            [
                1
                - 2
                * (self.y.clone() * self.y.clone() + self.z.clone() * self.z.clone()),
                2 * (self.x.clone() * self.y.clone() + self.z.clone() * self.w.clone()),
                2 * (self.x.clone() * self.z.clone() - self.y.clone() * self.w.clone()),
            ],
            dim=-1,
        )

    @property
    def xz_axis(self):
        # return torch.stack([
        #     torch.stack([1 - 2 * (self.y.pow(2) + self.z.pow(2)),
        #     2 * (self.x * self.y - self.z * self.w),
        #     2 * (self.x * self.z + self.y * self.w)]),
        #     torch.stack([2 * (self.x * self.z + self.y * self.w),
        #     2 * (self.y * self.z - self.x * self.w),
        #     1 - 2 * (self.x.pow(2) + self.y.pow(2))])
        # ]) # debug xzaxis format using clone like x_axis
        first_row = torch.stack(
            [
                1
                - 2
                * (self.y.clone() * self.y.clone() + self.z.clone() * self.z.clone()),
                2 * (self.x.clone() * self.y.clone() - self.z.clone() * self.w.clone()),
                2 * (self.x.clone() * self.z.clone() + self.y.clone() * self.w.clone()),
            ],
            dim=-1,
        )
        second_row = torch.stack(
            [
                2 * (self.x.clone() * self.z.clone() + self.y.clone() * self.w.clone()),
                2 * (self.y.clone() * self.z.clone() - self.x.clone() * self.w.clone()),
                1
                - 2
                * (self.x.clone() * self.x.clone() + self.y.clone() * self.y.clone()),
            ],
            dim=-1,
        )
        return torch.stack([first_row, second_row], dim=-2)

    @property
    def shape(self):
        return (self.w.shape[0], 4)

    @property
    def real(self):
        return self.w

    @property
    def imag(self):
        return torch.stack([self.x, self.y, self.z], dim=-1)

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            w = (
                self.w * other.w
                - self.x * other.x
                - self.y * other.y
                - self.z * other.z
            )
            x = (
                self.w * other.x
                + self.x * other.w
                + self.y * other.z
                - self.z * other.y
            )
            y = (
                self.w * other.y
                - self.x * other.z
                + self.y * other.w
                + self.z * other.x
            )
            z = (
                self.w * other.z
                + self.x * other.y
                - self.y * other.x
                + self.z * other.w
            )
            return Quaternion(w, x, y, z)
        elif isinstance(other, (int, float, torch.Tensor)):
            return Quaternion(
                self.w * other, self.x * other, self.y * other, self.z * other
            )
        else:
            raise ValueError("unsupported type")

    def __truediv__(self, other):
        if isinstance(other, (int, float, torch.Tensor)):
            return Quaternion(
                self.w / other, self.x / other, self.y / other, self.z / other
            )
        else:
            raise ValueError("unsupported type")

    def __add__(self, other):
        if isinstance(other, Quaternion):
            return Quaternion(
                self.w + other.w, self.x + other.x, self.y + other.y, self.z + other.z
            )
        elif isinstance(other, torch.Tensor):
            w, x, y, z = Quaternion._split_tensor_components(other)
            return Quaternion(self.w + w, self.x + x, self.y + y, self.z + z)
        else:
            raise ValueError("unsupported type")

    def __sub__(self, other):
        return Quaternion(
            self.w - other.w, self.x - other.x, self.y - other.y, self.z - other.z
        )

    def __neg__(self):
        return Quaternion(-self.w, -self.x, -self.y, -self.z)

    def __repr__(self):
        return f"({self.w}, {self.x}i, {self.y}j, {self.z}k)"

    def __getitem__(self, indices):
        return Quaternion(
            self.w[indices], self.x[indices], self.y[indices], self.z[indices]
        )

    def __setitem__(self, indices, value):
        if isinstance(value, Quaternion):
            # Assign directly for a single index
            self.w[indices] = value.w
            self.x[indices] = value.x
            self.y[indices] = value.y
            self.z[indices] = value.z
        elif isinstance(value, torch.Tensor):
            w, x, y, z = Quaternion._split_tensor_components(value)
            self.w[indices] = w
            self.x[indices] = x
            self.y[indices] = y
            self.z[indices] = z
        else:
            raise ValueError("Assigned value must be an instance of quaternion")

    def inverse(self):
        return self.conjugate() / self.norm()

    def norm(self):
        return torch.sqrt(self.w.pow(2) + self.x.pow(2) + self.y.pow(2) + self.z.pow(2))

    def normalize(self):
        return self / self.norm()

    def conjugate(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def toTensor(self):
        return torch.stack([self.w, self.x, self.y, self.z], dim=-1)

    def append(self, other):
        self.w = torch.cat([self.w, other.w])
        self.x = torch.cat([self.x, other.x])
        self.y = torch.cat([self.y, other.y])
        self.z = torch.cat([self.z, other.z])

    def toEuler(self, order="zyx"):
        if order == "zyx":
            roll = torch.atan2(
                2 * (self.w * self.x + self.y * self.z),
                1 - 2 * (self.x.pow(2) + self.y.pow(2)),
            )
            pitch = torch.asin(2 * (self.w * self.y - self.z * self.x))
            yaw = torch.atan2(
                2 * (self.w * self.z + self.x * self.y),
                1 - 2 * (self.y.pow(2) + self.z.pow(2)),
            )
            return torch.stack([roll, pitch, yaw], dim=-1)
        elif order == "xyz":
            roll = torch.atleast_1d(
                torch.atan2(
                    2 * (self.w * self.y - self.x * self.z),
                    1 - 2 * (self.x.pow(2) + self.y.pow(2)),
                )
            )
            pitch = torch.atleast_1d(
                torch.asin(2 * (self.w * self.z - self.y * self.x))
            )
            yaw = torch.atleast_1d(
                torch.atan2(
                    2 * (self.w * self.x + self.y * self.z),
                    1 - 2 * (self.x.pow(2) + self.z.pow(2)),
                )
            )
            return torch.stack([roll, pitch, yaw], dim=-1)

    @staticmethod
    def from_euler(roll, pitch, yaw, order="zyx"):
        roll, pitch, yaw = (
            torch.as_tensor(roll),
            torch.as_tensor(pitch),
            torch.as_tensor(yaw),
        )
        if order == "zyx":
            cy = torch.cos(yaw * 0.5)
            sy = torch.sin(yaw * 0.5)
            cp = torch.cos(pitch * 0.5)
            sp = torch.sin(pitch * 0.5)
            cr = torch.cos(roll * 0.5)
            sr = torch.sin(roll * 0.5)
            w = cr * cp * cy + sr * sp * sy
            x = sr * cp * cy - cr * sp * sy
            y = cr * sp * cy + sr * cp * sy
            z = cr * cp * sy - sr * sp * cy
        elif order == "xyz":
            cy = torch.cos(yaw * 0.5)
            sy = torch.sin(yaw * 0.5)
            cp = torch.cos(pitch * 0.5)
            sp = torch.sin(pitch * 0.5)
            cr = torch.cos(roll * 0.5)
            sr = torch.sin(roll * 0.5)
            w = cr * cp * cy - sr * sp * sy
            x = sr * cp * cy + cr * sp * sy
            y = cr * sp * cy - sr * cp * sy
            z = cr * cp * sy + sr * sp * cy
        return Quaternion(w, x, y, z)

    def clone(self):
        return Quaternion(
            self.w.clone(), self.x.clone(), self.y.clone(), self.z.clone()
        )

    def detach(self):
        return Quaternion(
            self.w.detach(), self.x.detach(), self.y.detach(), self.z.detach()
        )

    def __len__(self):
        try:
            return len(self.w)
        except TypeError:
            return 1


class Integrator:
    def __init__(self):
        pass

    @staticmethod
    def _get_derivatives(
        vel: torch.tensor,
        ori: torch.tensor,
        acc: torch.tensor,
        ori_vel: torch.tensor,
        tau: torch.tensor,
        J: torch.tensor,
        J_inv: torch.tensor,
    ):
        d_pos = vel
        omega = ori_vel
        zeros = torch.zeros(omega.shape[0], device=omega.device, dtype=omega.dtype)
        omega_quat = Quaternion(zeros, omega[:, 0], omega[:, 1], omega[:, 2])
        d_q = (ori * omega_quat * 0.5).toTensor()
        d_vel = acc
        tau_tensor = tau
        J_omega = omega @ J.T
        coriolis = torch.linalg.cross(omega, J_omega, dim=1)
        d_ori_vel = (tau_tensor - coriolis) @ J_inv.T
        return d_pos, d_q, d_vel, d_ori_vel

    @staticmethod
    def integrate(
        pos: torch.tensor,
        ori: torch.tensor,
        vel: torch.tensor,
        ori_vel: torch.tensor,
        acc: torch.tensor,
        tau: torch.tensor,
        J: torch.tensor,
        J_inv: torch.tensor,
        dt: torch.tensor,
        type="1st_order_euler",
    ):
        """
        Args:
            (pos, ori, vel, ori_vel): current state, shape (N, 3), (N, 4), (N, 3), (N, 3)
            acc: linear acceleration in world frame, shape (N, 3)
            tau: torque in body frame, shape (N, 3)
            J: inertia matrix in body frame, shape (3, 3)
                Future work: to be batched, add domain randomization
            J_inv: inverse of inertia matrix in body frame, shape (3, 3)
                Future work: to be batched, add domain randomization
            dt: time step, shape (1,) or scalar
            type: integration method, one of ['1st_order_euler', 'rk4']
        Returns:
            (pos, ori, vel, ori_vel): next state, shape (N, 3), (N, 4), (N, 3), (N, 3)
            d_ori_vel: angular acceleration in body frame, shape (N, 3)

        """
        if type == "1st_order_euler":
            _, ori_cache, vel_cache, ori_vel_cache = (
                pos.clone(),
                ori.clone(),
                vel.clone(),
                ori_vel.clone(),
            )

            d_pos, d_ori, d_vel, d_ori_vel = Integrator._get_derivatives(
                vel=vel_cache,
                ori=ori_cache,
                acc=acc,
                ori_vel=ori_vel_cache,
                tau=tau,
                J=J,
                J_inv=J_inv,
            )
            pos += d_pos * dt
            ori += d_ori * dt
            ori = ori / ori.norm()

            vel += d_vel * dt
            ori_vel += d_ori_vel * dt

            # ori = ori / ori.norm()

            return pos, ori, vel, ori_vel, d_ori_vel

        elif type == "rk4":
            ks = torch.tensor([1.0, 2.0, 2.0, 1.0]) / 6
            slice_ts = torch.tensor([0.5, 0.5, 1])
            _, ori_cache, vel_cache, ori_vel_cache = (
                pos.clone(),
                ori.clone(),
                vel.clone(),
                ori_vel.clone(),
            )
            d_pos = torch.zeros((pos.shape[0], pos.shape[1], 4))
            d_ori = torch.zeros((ori.shape[0], ori.shape[1], 4))
            d_vel = torch.zeros((vel.shape[0], vel.shape[1], 4))
            d_ori_vel = torch.zeros((ori_vel.shape[0], ori_vel.shape[1], 4))

            for index in range(4):
                # pos_cache = pos + d_pos * slice_ts[index] * dt
                if index != 0:
                    ori_cache = ori + d_ori[:, :, index - 1] * slice_ts[index - 1] * dt
                    vel_cache = vel + d_vel[:, :, index - 1] * slice_ts[index - 1] * dt
                    ori_vel_cache = (
                        ori_vel + d_ori_vel[:, :, index - 1] * slice_ts[index - 1] * dt
                    )

                (
                    d_pos[:, :, index],
                    d_ori[:, :, index],
                    d_vel[:, :, index],
                    d_ori_vel[:, :, index],
                ) = Integrator._get_derivatives(
                    vel=vel_cache,
                    ori=ori_cache,
                    acc=acc,
                    ori_vel=ori_vel_cache,
                    tau=tau,
                    J=J,
                    J_inv=J_inv,
                )
            # f"w_cache: {ori_vel_cache} quat:{ori_cache} d_ori:{d_ori[:,:,index]}"
            pos += d_pos @ ks * dt
            ori += d_ori @ ks * dt
            vel += d_vel @ ks * dt
            ori_vel += d_ori_vel @ ks * dt

            return pos, ori, vel, ori_vel, d_ori_vel

        else:
            raise ValueError("type should be one of ['euler', 'rk4']")

    @staticmethod
    def integrate_surrogate(
        pos: torch.tensor,
        ori: torch.tensor,
        vel: torch.tensor,
        ori_vel: torch.tensor,
        acc: torch.tensor,
        tau: torch.tensor,
        J: torch.tensor,
        J_inv: torch.tensor,
        dt: torch.tensor,
        type="1st_order_euler",
    ):
        """
        Args:
            (pos, ori, vel, ori_vel): current state, shape (N, 3), (N, 4), (N, 3), (N, 3)
            acc: linear acceleration in world frame, shape (N, 3)
            tau: torque in body frame, shape (N, 3)
            J: inertia matrix in body frame, shape (3, 3)
                Future work: to be batched, add domain randomization
            J_inv: inverse of inertia matrix in body frame, shape (3, 3)
                Future work: to be batched, add domain randomization
            dt: time step, shape (1,) or scalar
            type: integration method, one of ['1st_order_euler', 'rk4']
        Returns:
            (pos, ori, vel, ori_vel): next state, shape (N, 3), (N, 4), (N, 3), (N, 3)
            d_ori_vel: angular acceleration in body frame, shape (N, 3)

        """
        if type == "1st_order_euler":
            _, ori_cache, vel_cache, ori_vel_cache = (
                pos.clone(),
                ori.clone(),
                vel.clone(),
                ori_vel.clone(),
            )

            d_pos, d_ori, d_vel, d_ori_vel = Integrator._get_derivatives(
                vel=vel_cache,
                ori=ori_cache,
                acc=acc,
                ori_vel=ori_vel_cache,
                tau=tau,
                J=J,
                J_inv=J_inv,
            )
            pos += d_pos * dt
            ori += d_ori * dt
            ori = ori / ori.norm()

            vel += d_vel * dt
            ori_vel += d_ori_vel * dt

            # ori = ori / ori.norm()

            return pos, ori, vel, ori_vel, d_ori_vel


def cross(a: torch.Tensor, b: torch.Tensor):
    res = (
        torch.stack(
            [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ]
        )
        + 0
    )
    return res


def debug():
    pass


if __name__ == "__main__":
    debug()
