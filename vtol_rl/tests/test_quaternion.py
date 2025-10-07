import torch
from scipy.spatial.transform import Rotation

from vtol_rl.utils.maths import Quaternion


def test_quaternion_tensor_init():
    quat = torch.tensor(
        [
            [0.7071, 0.0, 0.7071, 0.0],
            [0.7071, 0.7071, 0.0, 0.0],
        ]
    )
    qq = Quaternion(quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3])


def test_quaternion_batch_shapes():
    batch = 4
    quat = Quaternion(num=batch)

    assert quat.toTensor().shape == (batch, 4)
    assert quat.R.shape == (batch, 3, 3)
    assert quat.x_axis.shape == (batch, 3)
    assert quat.toEuler().shape == (batch, 3)


def test_quaternion_rotate_accepts_batch_and_column_vectors():
    roll = torch.zeros(2)
    pitch = torch.zeros(2)
    yaw = torch.tensor([0.0, torch.pi / 2])

    quat = Quaternion.from_euler(roll, pitch, yaw)

    vectors = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )

    rotated = quat.rotate(vectors)
    assert rotated.shape == (2, 3)
    assert torch.allclose(rotated[0], vectors[0])
    assert torch.allclose(rotated[1], torch.tensor([0.0, 1.0, 0.0]))

    column_form = vectors.T
    rotated_columns = quat.rotate(column_form)
    assert rotated_columns.shape == (3, 2)
    assert torch.allclose(rotated_columns, rotated.T)


def test_quaternion_inv_rotate_inverts_rotation():
    quat = Quaternion.from_euler(0.0, 0.0, torch.pi / 2)
    vector = torch.tensor([1.0, 0.0, 0.0])

    rotated = quat.rotate(vector)
    recovered = quat.inv_rotate(rotated)

    assert torch.allclose(rotated, torch.tensor([0.0, 1.0, 0.0]))
    assert torch.allclose(recovered, vector)


def test_quaternion_matches_scipy_rotation():
    batch = 5
    roll = torch.linspace(-0.3, 0.3, batch)
    pitch = torch.linspace(0.1, -0.2, batch)
    yaw = torch.linspace(0.05, -0.4, batch)

    quat = Quaternion.from_euler(roll, pitch, yaw)
    scipy_rot = Rotation.from_euler(
        "xyz", torch.stack([roll, pitch, yaw], dim=1).numpy()
    )

    torch_R = quat.R
    scipy_R = torch.from_numpy(scipy_rot.as_matrix()).to(dtype=torch_R.dtype)
    assert torch.allclose(torch_R, scipy_R, atol=1e-6, rtol=1e-6)

    scipy_quat_xyzw = torch.from_numpy(scipy_rot.as_quat()).to(torch_R.dtype)
    scipy_quat_wxyz = torch.column_stack(
        (
            scipy_quat_xyzw[:, 3],
            scipy_quat_xyzw[:, 0],
            scipy_quat_xyzw[:, 1],
            scipy_quat_xyzw[:, 2],
        )
    )

    ours = quat.toTensor()
    alignment = torch.sign((ours * scipy_quat_wxyz).sum(dim=1, keepdim=True))
    alignment[alignment == 0] = 1
    scipy_quat_aligned = scipy_quat_wxyz * alignment
    assert torch.allclose(ours, scipy_quat_aligned, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    test_quaternion_tensor_init()
    test_quaternion_batch_shapes()
    test_quaternion_rotate_accepts_batch_and_column_vectors()
    test_quaternion_inv_rotate_inverts_rotation()
    test_quaternion_matches_scipy_rotation()
    print("All tests passed!")
