import torch

from nonprehensile.bank import build
from nonprehensile.env import EnvConfig, TrayEnv
from nonprehensile.math import mv, rotation


def test_accelerating_frame_preserves_free_fall(tmp_path):
    torch.set_num_threads(2)
    bank = tmp_path / "bank.npz"
    build(bank, count=4, workers=1)
    env = TrayEnv(EnvConfig(num_envs=2), bank, "cpu")
    env.sim.data.qpos[:, :3] = torch.tensor([0.0, 0.0, 1.0])
    env.sim.forward()
    chi = torch.tensor([[0.2, -0.3, 0.4, 0.3, 0.2, -0.1]]).expand(2, 6)
    env.physics(chi)
    rot = rotation(env.quaternion)
    c, _, _, wr, _, _ = env.state()
    world_com = env.position + mv(rot, c)
    expected = torch.tensor([[0.0, 0.0, 1.0 - 0.5 * 9.81 * 0.02**2]]).expand(2, 3)
    torch.testing.assert_close(world_com, expected, atol=3e-4, rtol=0)
    torch.testing.assert_close(mv(rot, env.omega + wr), torch.zeros(2, 3), atol=1e-3, rtol=0)


def test_all_reward_modes_step_with_randomized_masses(tmp_path):
    torch.set_num_threads(2)
    bank = tmp_path / "bank.npz"
    build(bank, count=4, workers=1)
    for task in ("sticking", "sliding", "tipping", "twisting"):
        env = TrayEnv(EnvConfig(task=task, group="E", num_envs=2), bank, "cpu")
        obs, reward, done, info = env.step(torch.zeros(2, 6))
        assert obs.shape == (2, 46)
        assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
        assert not done.any()
        assert torch.isfinite(info["distance"]).all()
        torch.testing.assert_close(env.sim.model.body_mass[:, 1], env.mass)
