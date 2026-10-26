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
    expected = torch.tensor([[0.0, 0.0, 1.0 - 0.5 * 9.81 * env.dt**2]]).expand(2, 3)
    torch.testing.assert_close(world_com, expected, atol=3e-4, rtol=0)
    torch.testing.assert_close(mv(rot, env.omega + wr), torch.zeros(2, 3), atol=1e-3, rtol=0)


def test_all_reward_modes_step_with_fixed_mass_and_randomized_inertia(tmp_path):
    torch.set_num_threads(2)
    bank = tmp_path / "bank.npz"
    build(bank, count=4, workers=1)
    for task in ("sticking", "sliding", "tipping", "twisting"):
        env = TrayEnv(EnvConfig(task=task, group="E", num_envs=2), bank, "cpu")
        obs, reward, done, info = env.step(torch.zeros(2, 6))
        assert obs.shape == (2, 36)
        assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
        assert not done.any()
        assert torch.isfinite(info["distance"]).all()
        torch.testing.assert_close(env.sim.model.body_mass[:, 1], env.mass)
        torch.testing.assert_close(env.mass, torch.ones(2))
        assert (info["min_normal_load"] > 0.1).all()


def test_observations_and_mapping_failure(tmp_path):
    bank = tmp_path / "bank.npz"
    build(bank, count=2, workers=1)
    env = TrayEnv(EnvConfig(task="sliding", group="F", num_envs=2), bank, "cpu")
    env.quaternion[:] = torch.tensor([2**-0.5, 0, 0, 2**-0.5])
    env.velocity[:] = torch.tensor([0.0, 1.0, 0.0])
    obs = env.observe()
    torch.testing.assert_close(obs[:, :3], env.sim.data.qpos[:, :3] / 0.4)
    torch.testing.assert_close(
        obs[:, 18:21], torch.tensor([[0.5, 0.0, 0.0]]).expand(2, 3), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(obs[:, 24:27], torch.tensor([[0.0, 0.0, -1.0]]).expand(2, 3))
    torch.testing.assert_close(obs[:, 31:34], torch.tensor([[1.0, 0.0, 0.0]]).expand(2, 3))
    assert ((obs[:, 27:31] >= -1) & (obs[:, 27:31] <= 1)).all()
    # A failed map is terminal, logged independently of physical failure, then reset.
    obs, reward, done, info = env.step(torch.full((2, 6), torch.nan), torch.ones(2).bool())
    assert done.all() and info["mapping_failure"].all()
    assert not info["success"].any() and not info["physical_failure"].any()
    assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
    assert (env.steps == 0).all()
