"""CPU tensor regressions for settings and event ordering; no simulator fallback."""

from types import SimpleNamespace

import pytest
import torch

from nonprehensile.env import EnvConfig, TrayEnv
from nonprehensile.math import rotation


def tensor_env(task, n=1):
    env = TrayEnv.__new__(TrayEnv)
    env.cfg = EnvConfig(task=task, num_envs=n)
    env.device, env.n = "cpu", n
    env.entered = torch.zeros(n, dtype=torch.bool)
    env.steps = torch.zeros(n, dtype=torch.long)
    env.position, env.velocity, env.omega, env.origin = [env.zeros(3) for _ in range(4)]
    env.quaternion = torch.tensor([[1.0, 0, 0, 0]]).repeat(n, 1)
    env.size = torch.tensor([[0.2, 0.2, 0.2]]).repeat(n, 1)
    env.mass = torch.ones(n)
    c, r = env.zeros(3), torch.eye(3).repeat(n, 1, 1)
    c[:, 2] = 0.1
    env.state = lambda: (c, r, env.zeros(3), env.zeros(3), env.zeros(3), r)
    return env, c, r


@pytest.mark.parametrize("exceeded", [None, "translation", "rotation", "speed", "spin"])
def test_relaxed_limits_still_reject_excesses(exceeded):
    env, c, r = tensor_env("sticking")
    c[:, 0] = 0.006 if exceeded != "translation" else 0.011
    angle = torch.deg2rad(torch.tensor(5.0 if exceeded != "rotation" else 11.0))
    # Yaw keeps the bottom face on the tray, isolating the attitude tolerance.
    q = torch.tensor([[torch.cos(angle / 2), 0, 0, torch.sin(angle / 2)]])
    r[:] = rotation(q)
    env.velocity[:, 0] = 3 if exceeded != "speed" else 5.1
    env.omega[:, 0] = 5 if exceeded != "spin" else 10.1
    _, failed, _, _ = env.outcomes()
    assert failed.item() == (exceeded is not None)


def test_tipping_first_event_survives_rebound_but_not_prior_failure():
    env, _, _ = tensor_env("tipping", n=3)
    counter = [0]

    def step():
        counter[0] += 1

    env.sim = SimpleNamespace(data=SimpleNamespace(qfrc_applied=env.zeros(6)), step=step)

    def outcomes():
        k = counter[0]
        failed = torch.tensor([k >= 8, 2 <= k < 4, False])
        success = torch.tensor([4 <= k < 8, k >= 4, False])
        return success, failed, torch.ones(3, dtype=torch.bool), torch.full((3,), float(k))

    env.outcomes = outcomes
    success, failure, started, error, duration = env.physics(env.zeros(6))
    assert success.tolist() == [True, False, False]
    assert failure.tolist() == [False, True, False]
    assert started.all()
    torch.testing.assert_close(duration, torch.tensor([0.004, 0.002, 0.02]))
    torch.testing.assert_close(error, torch.tensor([4.0, 2.0, 20.0]))
