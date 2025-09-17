import torch


class Quaternion:
    def __init__(
        self, w=None, x=None, y=None, z=None, num=1, device=torch.device("cpu")
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
        elif other.shape[0] == 3:
            # vector rotation
            return (self * Quaternion(torch.tensor(0), *other) * self.conjugate()).imag

    def inv_rotate(self, other):
        """
        rotate to local
        """
        if isinstance(other, Quaternion):
            # quaternion multiplication
            return self.conjugate() * other
        elif other.shape[0] == 3:
            # vector rotation
            return (self.conjugate() * Quaternion(torch.tensor(0), *other) * self).imag

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
        # return torch.permute(torch.stack([
        #     torch.stack([1 - 2 * (self.y.pow(2) + self.z.pow(2)), 2 * (self.x * self.y - self.z * self.w), 2 * (self.x * self.z + self.y * self.w)]),
        #     torch.stack([2 * (self.x * self.y + self.z * self.w), 1 - 2 * (self.x.pow(2) + self.z.pow(2)), 2 * (self.y * self.z - self.x * self.w)]),
        #     torch.stack([2 * (self.x * self.z - self.y * self.w), 2 * (self.y * self.z + self.x * self.w), 1 - 2 * (self.x.pow(2) + self.y.pow(2))])
        # ]), (2,0,1))
        return torch.stack(
            [
                torch.stack(
                    [
                        1 - 2 * (self.y.pow(2) + self.z.pow(2)),
                        2 * (self.x * self.y - self.z * self.w),
                        2 * (self.x * self.z + self.y * self.w),
                    ]
                ),
                torch.stack(
                    [
                        2 * (self.x * self.y + self.z * self.w),
                        1 - 2 * (self.x.pow(2) + self.z.pow(2)),
                        2 * (self.y * self.z - self.x * self.w),
                    ]
                ),
                torch.stack(
                    [
                        2 * (self.x * self.z - self.y * self.w),
                        2 * (self.y * self.z + self.x * self.w),
                        1 - 2 * (self.x.pow(2) + self.y.pow(2)),
                    ]
                ),
            ]
        )

    @property
    def x_axis(self):
        # return torch.stack([1 - 2 * (self.y.pow(2) + self.z.pow(2)), 2 * (self.x * self.y + self.z * self.w), 2 * (self.x * self.z - self.y * self.w)])
        x_axis = torch.stack(
            [
                # 1 - 2 * (self.y*self.y + self.z*self.z),
                #  2 * (self.x * self.y + self.z * self.w),
                # 2 * (self.x * self.z - self.y * self.w)
                1
                - 2
                * (self.y.clone() * self.y.clone() + self.z.clone() * self.z.clone()),
                2 * (self.x.clone() * self.y.clone() + self.z.clone() * self.w.clone()),
                2 * (self.x.clone() * self.z.clone() - self.y.clone() * self.w.clone()),
            ]
        )
        return x_axis

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
        return torch.stack(
            [
                torch.stack(
                    [
                        1
                        - 2
                        * (
                            self.y.clone() * self.y.clone()
                            + self.z.clone() * self.z.clone()
                        ),
                        2
                        * (
                            self.x.clone() * self.y.clone()
                            - self.z.clone() * self.w.clone()
                        ),
                        2
                        * (
                            self.x.clone() * self.z.clone()
                            + self.y.clone() * self.w.clone()
                        ),
                    ]
                ),
                torch.stack(
                    [
                        2
                        * (
                            self.x.clone() * self.z.clone()
                            + self.y.clone() * self.w.clone()
                        ),
                        2
                        * (
                            self.y.clone() * self.z.clone()
                            - self.x.clone() * self.w.clone()
                        ),
                        1
                        - 2
                        * (
                            self.x.clone() * self.x.clone()
                            + self.y.clone() * self.y.clone()
                        ),
                    ]
                ),
            ]
        )

    @property
    def shape(self):
        return 4, len(self)

    @property
    def real(self):
        return self.w

    @property
    def imag(self):
        return torch.stack([self.x, self.y, self.z])

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
            return Quaternion(
                self.w + other[0],
                self.x + other[1],
                self.y + other[2],
                self.z + other[3],
            )
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
            # Assign directly for a single index

            self.w[indices] = value[0]
            self.x[indices] = value[1]
            self.y[indices] = value[2]
            self.z[indices] = value[3]
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
        return torch.stack([self.w, self.x, self.y, self.z])

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
            return torch.stack([roll, pitch, yaw])  # roll pitch yaw
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
            return torch.stack([roll, pitch, yaw])  # roll pitch yaw

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
        d_q = (ori * Quaternion(torch.tensor(0), *ori_vel) * 0.5).toTensor()
        d_vel = acc
        # d_ori_vel = J_inv @ (tau - ori_vel.cross(J @ ori_vel))
        d_ori_vel = J_inv @ (tau - torch.linalg.cross(ori_vel.T, (J @ ori_vel).T).T)
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
        type="euler",
    ):
        if type == "euler":
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
