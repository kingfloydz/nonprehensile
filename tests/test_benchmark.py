import numpy as np
import pytest
import torch

from nonprehensile.bank import build
from nonprehensile.env import EnvConfig, TrayEnv
from nonprehensile.pipeline import configuration
from nonprehensile.settings import BENCHMARK_VERSION, SETTINGS


def test_banks_use_separate_mode_domains_and_settled_unit_mass(tmp_path):
    path = tmp_path / "bank.npz"
    build(path, count=8, workers=1)
    with np.load(path) as bank:
        assert str(bank["benchmark_version"]) == BENCHMARK_VERSION
        for task, spec in SETTINGS.items():
            size, mu = bank[f"{task}_size"], bank[f"{task}_mu"]
            assert np.all(size >= np.array(spec.size_low) - 1e-7)
            assert np.all(size <= np.array(spec.size_high) + 1e-7)
            assert np.all(mu >= spec.friction[0]) and np.all(mu <= spec.friction[1])
            assert np.isfinite(bank[f"{task}_qpos"]).all()
            np.testing.assert_allclose(np.linalg.norm(bank[f"{task}_quaternion"], axis=1), 1)
        assert bank["twisting_mu"].max() <= 0.12


def test_old_banks_fail_explicitly(tmp_path):
    path = tmp_path / "old.npz"
    np.savez(path, size=np.ones((1, 3)))
    with pytest.raises(ValueError, match="bank"):
        TrayEnv(EnvConfig(num_envs=1), path, "cpu")


@pytest.mark.parametrize("task", SETTINGS)
def test_time_and_goals_match_for_all_methods(task):
    for group in "ABCDEF":
        cfg = configuration(task, group, 2, 1, seed=3)
        env, ppo = cfg["env"], cfg["ppo"]
        assert env["seed"] == 3
        assert env["goal"] == SETTINGS[task].goal
        assert tuple(env["action_limits"]) == SETTINGS[task].action_limits
        assert ppo["rollout_steps"] * env["control_dt"] == 0.64
        assert ppo["gamma"] ** 4 == pytest.approx(0.99)
    assert EnvConfig(task="tipping").goal == pytest.approx(np.deg2rad(80))
    assert EnvConfig(task="twisting").goal == pytest.approx(np.deg2rad(45))


def test_vertical_twisting_action_is_not_replaced_by_a_script(tmp_path):
    bank = tmp_path / "bank.npz"
    build(bank, count=2, workers=1)
    env = TrayEnv(EnvConfig(task="twisting", num_envs=1), bank, "cpu")
    action = torch.zeros(1, 6)
    action[:, 5] = -0.5  # -1 m/s², irrespective of sampled mu
    _, _, _, info = env.step(action)
    torch.testing.assert_close(info["terminal_vertical_velocity"], torch.tensor([-0.005]))
