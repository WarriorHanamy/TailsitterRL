# 动力学系统文档

## 概述

本文档解释了 `vtol_rl/envs/base/dynamics.py` 中动力学系统的设计，该系统实现了垂直起降（VTOL）飞行器，特别是尾坐式无人机的物理仿真。

## 状态设计

### 状态向量结构

系统使用在 `vtol_rl/utils/type.py` 中定义的13维状态向量：

```python
STATE_POS = slice(0, 3)      # 位置 (x, y, z)
STATE_ORI = slice(3, 7)      # 姿态四元数 (qw, qx, qy, qz)
STATE_VEL = slice(7, 10)     # 线速度 (vx, vy, vz)
STATE_ANG_VEL = slice(10, 13) # 角速度 (wx, wy, wz)
```

### 核心状态变量

位于 `dynamics.py:34-45`：

- **_position**: 世界坐标系中的位置，形状 `(N, 3)`
- **_orientation**: 使用自定义 `Quaternion` 类的姿态四元数
- **_velocity**: 世界坐标系中的线速度，形状 `(N, 3)`
- **_angular_velocity**: 机体坐标系中的角速度，形状 `(N, 3)`
- **_acc**: 线加速度，形状 `(N, 3)`
- **_angular_acc**: 角加速度，形状 `(N, 3)`

### 闭环动力学状态变量（新增）

- **_thrust_state**: 推力闭环状态，形状 `(N, 1)`
- **_angular_velocity_state**: 角速度变化率状态，形状 `(N, 3)`

### 辅助状态变量

- **_thrusts**: 电机推力，形状 `(N, 4)`
- **_motor_omega**: 电机角速度，形状 `(N, 4)`
- **_t**: 仿真时间，形状 `(N,)`

## 系统状态设计

### 观测状态 (Observed State) - 13维
用于RL智能体观测，对应原来的13维状态向量：

```python
@property
def observed_state(self):
    return torch.cat([
        self.position,          # 位置 (0:3)
        self.orientation,       # 姿态四元数 (3:7)
        self.velocity,          # 线速度 (7:10)
        self.angular_velocity   # 角速度 (10:13)
    ], dim=1)  # 总维度: 13
```

### 完整状态 (Full State) - 17维
包含闭环动力学的内部状态，用于动力学仿真：

```python
@property
def full_state(self):
    return torch.cat([
        self.position,              # 位置 (0:3)
        self.orientation,           # 姿态四元数 (3:7)
        self.velocity,              # 线速度 (7:10)
        self.angular_velocity,      # 角速度 (10:13)
        self.thrust_state,          # 推力状态 (13:14)
        self.angular_velocity_state # 角速度变化率状态 (14:17)
    ], dim=1)  # 总维度: 17
```

### 状态切片定义（更新）
```python
# 观测状态切片（13维）
STATE_POS = slice(0, 3)          # 位置
STATE_ORI = slice(3, 7)          # 姿态四元数
STATE_VEL = slice(7, 10)         # 线速度
STATE_ANG_VEL = slice(10, 13)    # 角速度

# 完整状态切片（17维）
STATE_POS = slice(0, 3)          # 位置
STATE_ORI = slice(3, 7)          # 姿态四元数
STATE_VEL = slice(7, 10)         # 线速度
STATE_ANG_VEL = slice(10, 13)    # 角速度
STATE_THRUST = slice(13, 14)     # 推力状态
STATE_ANG_VEL_DOT = slice(14, 17) # 角速度变化率状态
```

## 控制量设计

### 动作空间 (Action Space) - 4维
RL智能体输出的控制指令：

```python
action = [
    target_thrust,        # 目标推力 (标量)
    target_omega_x,       # 目标角速度x (rad/s)
    target_omega_y,       # 目标角速度y (rad/s)
    target_omega_z        # 目标角速度z (rad/s)
]  # 总维度: 4
```

### 控制指令映射
```python
# 解析动作指令
target_thrust = action[:, 0:1]           # 形状 (N, 1)
target_angular_velocity = action[:, 1:4]  # 形状 (N, 3)
```

## 闭环动力学设计

### 设计思路

将当前的开环+PID控制设计改为闭环动力学，假设下游已完美实现推力和角速度控制：

```python
# 当前设计（开环 + PID控制）
action = [推力指令, 角速度_x, 角速度_y, 角速度_z]  # 归一化值
→ PID控制器 → 电机推力 → 物理仿真

# 闭环动力学设计
action = [目标推力, 目标角速度x, 目标角速度y, 目标角速度z]  # 物理量
→ 闭环动力学 → 物理仿真
```

### 推力闭环动力学（一阶模型）

#### 推力状态与指令分离
- **target_thrust**: RL动作输入的目标推力指令
- **thrust_state**: 推力闭环系统的内部状态
- 两者通过一阶动力学关联

#### 推力动力学方程
```python
# 一阶动力学模型：dT/dt = (target_thrust - T) / time_constant_T
# 离散化：T[k+1] = T[k] + (target_thrust - T[k]) * (dt / time_constant_T)
thrust_derivative = (target_thrust - self.thrust_state) / time_constant_T
self.thrust_state = self.thrust_state + thrust_derivative * dt
```

### 角速度闭环动力学（二阶模型）

#### 二阶动力学方程
```python
# 标准二阶系统：d²ω/dt² + 2ζω_n(dω/dt) + ω_n²ω = ω_n²ω_target
# 状态空间表示：
# x1 = ω (角速度)
# x2 = dω/dt (角速度变化率)
# dx1/dt = x2
# dx2/dt = ω_n²(ω_target - x1) - 2ζω_n * x2

# 计算角速度变化率的导数
omega_error = target_angular_velocity - self.angular_velocity

# x,y轴：欠阻尼系统 (ζ_xy = 0.7, ω_n_xy = 100.0)
zeta_xy = torch.tensor([0.7, 0.7], device=self.device).unsqueeze(0)  # (1, 2)
omega_n_xy = torch.tensor([100.0, 100.0], device=self.device).unsqueeze(0)  # (1, 2)

# z轴：过阻尼系统 (ζ_z = 1.5, ω_n_z = 8.0)
zeta_z = torch.tensor([8], device=self.device).unsqueeze(0)  # (1, 1)
omega_n_z = torch.tensor([50.0], device=self.device).unsqueeze(0)  # (1, 1)

# 合并阻尼参数
zeta = torch.cat([zeta_xy, zeta_z], dim=1)  # (1, 3)
omega_n = torch.cat([omega_n_xy, omega_n_z], dim=1)  # (1, 3)

# 计算角加速度
angular_acceleration = (
    omega_n**2 * omega_error
    - 2 * zeta * omega_n * self.angular_velocity_state
)

# 更新角速度状态
self.angular_velocity_state = self.angular_velocity_state + angular_acceleration * dt
self.angular_velocity = self.angular_velocity + self.angular_velocity_state * dt
```

### 完整状态向量总结

#### 状态维度对比
| 状态类型 | 维度 | 用途 | 包含内容 |
|---------|------|------|----------|
| observed_state | 13 | RL智能体观测 | 位置、姿态、速度、角速度 |
| full_state | 17 | 动力学仿真 | observed_state + 推力状态 + 角速度变化率状态 |

#### 状态变量列表
1. **位置** (3维): `[x, y, z]` - 世界坐标系
2. **姿态四元数** (4维): `[qw, qx, qy, qz]`
3. **线速度** (3维): `[vx, vy, vz]` - 世界坐标系
4. **角速度** (3维): `[wx, wy, wz]` - 机体坐标系
5. **推力状态** (1维): `[thrust_state]` - 闭环动力学内部状态
6. **角速度变化率状态** (3维): `[dωx/dt, dωy/dt, dωz/dt]` - 闭环动力学内部状态

#### 控制量列表
1. **目标推力** (1维): `[target_thrust]` - 物理量单位
2. **目标角速度** (3维): `[target_ωx, target_ωy, target_ωz]` - rad/s

## 线加速度计算

线加速度保持机体到世界的转换形式，包含阻力效应：

### 线加速度 = 推力加速度 + 重力 + 阻力加速度 + 噪声

#### 1. 推力分量（闭环动力学）
```python
# 推力指令到实际推力的闭环动力学（一阶模型）
target_thrust = action[0]  # 下游指令
# 一阶动力学模型：dT/dt = (target_thrust - T) / time_constant_T
thrust_state = thrust_state_prev + (target_thrust - thrust_state_prev) * (dt / time_constant_T)

# 机体坐标系推力
thrust_body = Dynamics.z * thrust_state.unsqueeze(1)
```
- `Dynamics.z = [0, 0, 1]` （机体Z轴方向）
- `thrust_state` 是推力闭环动力学的内部状态
- `time_constant_T` 是推力响应的时间常数，控制推力变化的快慢
- 一阶模型模拟了推力系统的惯性特性，推力变化不会瞬间完成

#### 2. 阻力分量
```python
velocity_body = self._orientation.inv_rotate(self._velocity)  # 转换到机体坐标系
linear_drag = linear_drag_coeffs * velocity_body
quadratic_drag = quadratic_drag_coeffs * velocity_body * velocity_body.abs()
drag = linear_drag + quadratic_drag
```
- **线性阻力**: 与速度成正比
- **二次阻力**: 与速度平方成正比

#### 3. 净力和加速度（body-to-world转换）
```python
proper_force_body = thrust_body - drag  # 机体坐标系净力
self._acc = self._orientation.rotate(proper_force_body) / self.m + gravity
```
- `self._orientation.rotate(proper_force_body) / self.m`: 推力和阻力产生的加速度（body-to-world转换）
- `gravity = Dynamics.g = [0, 0, -9.81]`: 重力加速度

#### 4. 过程噪声
```python
acc_noise = torch.randn_like(self._acc) * 0.5
```

### 完整公式
```
线加速度 = (R_机体到世界 × (推力状态_机体 - 阻力_机体)) / 质量 + 重力_世界 + 噪声
```

其中：
- `R_机体到世界`: 从机体坐标系到世界坐标系的旋转矩阵
- `推力状态_机体 = [0, 0, thrust_state]` （推力闭环动力学状态）
- `阻力_机体 = 线性阻力系数 × 速度_机体 + 二次阻力系数 × |速度_机体| × 速度_机体`
- `重力_世界 = [0, 0, -9.81] m/s²`

### 闭环动力学的优势

1. **接口清晰**: 输入直接对应物理量指令，便于理解
2. **计算效率**: 省去PID控制器和电机动力学计算
3. **稳定性**: 避免PID调参问题，假设下游已完美控制
4. **通用性**: 不依赖特定的控制器实现
5. **模块化**: 控制逻辑与动力学完全分离
6. **真实感**: 通过推力闭环动力学保留控制系统的动态特性

