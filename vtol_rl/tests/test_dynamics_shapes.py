import pytest
import torch

from vtol_rl.envs.base.dynamics import Dynamics


def test_dynamics_reset_rejects_feature_first_inputs():
    dyn = Dynamics(num=2)
    feature_first_vec3 = torch.zeros((3, dyn.num))
    feature_first_vec4 = torch.zeros((4, dyn.num))
    feature_first_scalar = torch.zeros((1, dyn.num))

    with pytest.raises(ValueError):
        dyn.reset(pos=feature_first_vec3)
    with pytest.raises(ValueError):
        dyn.reset(vel=feature_first_vec3)
    with pytest.raises(ValueError):
        dyn.reset(ori=feature_first_vec4)
    with pytest.raises(ValueError):
        dyn.reset(ori_vel=feature_first_vec3)
    with pytest.raises(ValueError):
        dyn.reset(motor_omega=feature_first_vec4)
    with pytest.raises(ValueError):
        dyn.reset(thrusts=feature_first_vec4)
    with pytest.raises(ValueError):
        dyn.reset(t=feature_first_scalar)


def test_dynamics_reset_accepts_batch_major_inputs():
    dyn = Dynamics(num=3)
    pos = torch.randn((dyn.num, 3))
    vel = torch.randn((dyn.num, 3))
    ori_vel = torch.randn((dyn.num, 3))
    ori = torch.zeros((dyn.num, 4))
    ori[:, 0] = 1
    motor_speed = torch.full((dyn.num, 4), dyn._init_motor_omega)
    thrusts = torch.full((dyn.num, 4), dyn._init_thrust_mag)
    times = torch.zeros(dyn.num)

    dyn.reset(
        pos=pos,
        vel=vel,
        ori_vel=ori_vel,
        ori=ori,
        motor_omega=motor_speed,
        thrusts=thrusts,
        t=times,
    )

    torch.testing.assert_close(dyn.position, pos)
    torch.testing.assert_close(dyn.velocity, vel)
    torch.testing.assert_close(dyn.angular_velocity, ori_vel)
    torch.testing.assert_close(dyn.thrusts, thrusts)
    torch.testing.assert_close(dyn.motor_omega, motor_speed)
    torch.testing.assert_close(dyn.t, times)


def test_dynamics_reset_rejects_mismatched_batch_with_indices():
    dyn = Dynamics(num=3)
    indices = torch.tensor([0])
    pos = torch.zeros((2, 3))

    with pytest.raises(ValueError, match="batch dimension must match"):
        dyn.reset(pos=pos, indices=indices)


def test_dynamics_reset_accepts_time_column_vector():
    dyn = Dynamics(num=2)
    times = torch.zeros((dyn.num, 1))

    dyn.reset(t=times)

    torch.testing.assert_close(dyn.t, torch.zeros(dyn.num))
