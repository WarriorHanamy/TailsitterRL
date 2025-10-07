import torch

from vtol_rl.envs.base.dynamics import Dynamics
from vtol_rl.envs.HoverEnv import HoverEnv
from vtol_rl.utils.type import Uniform


def test_dynamics():
    dyn = Dynamics(num=2)
    assert dyn.position.shape == (2, 3)
    assert dyn.orientation.shape == (2, 4)
    assert dyn.velocity.shape == (2, 3)
    assert dyn.angular_velocity.shape == (2, 3)
    assert dyn.state.shape == (2, 13)
    assert dyn.full_state.shape == (2, 17)
    dyn.reset()
    assert dyn.state.shape == (2, 13)
    assert dyn.full_state.shape == (2, 17)
    tor_tensor = torch.Tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
        ]
    )
    dyn._de_normalize(tor_tensor)


def test_hover_env():
    num_scene = 10
    env = HoverEnv(
        num_agent_per_scene=1,
        num_scene=num_scene,
        device="cpu",
        max_episode_steps=100,
        tensor_output=True,
        visual=False,
    )
    # assert env type is VecEnv
    obs = env.reset()
    assert obs["state"].shape == (num_scene, 13)
    single_action = torch.tensor([0.0, 0.0, 0.0, 0.0])
    action = torch.ones((num_scene, 4)) * single_action  # Hadamard Broadcasting

    # info is false; weird. REC MARK
    for i in range(10):
        obs, reward, done, info = env.step(action)

    assert obs["state"].shape == (num_scene, 13)
    assert reward.shape == (num_scene,)
    assert done.shape == (num_scene,)
    assert isinstance(info, list) and len(info) == num_scene
    env.close()


def test_uniform_gaussian():
    t = Uniform(2.0, 1.0)
    mm = t.sample(size=10)
    print(f"mm.shape: {mm.shape},\n {mm}")
    print(f"t.mean: {t.mean}, t.radius: {t.radius}")

    aa = t.per_sample_normalize(mm)
    print(f"aa.shape: {aa.shape},\n {aa}")


def main():
    single_action = torch.tensor([0.5, 0.0, 0.0, 0.0])
    action = torch.ones((10, 4)) * single_action
    print(f"action.shape: {action.shape},\n {action}")


if __name__ == "__main__":
    test_hover_env()
    # test_uniform_gaussian()
