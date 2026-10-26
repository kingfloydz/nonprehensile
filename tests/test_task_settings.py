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
    env.normal_fraction = torch.ones(n)
    env.load_history = torch.ones(n, 40)
    env.load_cursor = 0
    c, r = env.zeros(3), torch.eye(3).repeat(n, 1, 1)
    c[:, 2] = 0.1
    env.state = lambda: (c, r, env.zeros(3), env.zeros(3), env.zeros(3), r)
    return env, c, r


@pytest.mark.parametrize("exceeded", [None, "translation", "rotation", "speed", "spin"])
def test_mode_and_speed_limits_reject_excesses(exceeded):
    env, c, r = tensor_env("sticking")
    c[:, 0] = 0.0006 if exceeded != "translation" else 0.0011
    angle = torch.deg2rad(torch.tensor(0.5 if exceeded != "rotation" else 1.1))
    # Yaw keeps the bottom face on the tray, isolating the attitude tolerance.
    q = torch.tensor([[torch.cos(angle / 2), 0, 0, torch.sin(angle / 2)]])
    r[:] = rotation(q)
    env.velocity[:, 0] = 1 if exceeded != "speed" else 1.6
    env.omega[:, 0] = 1 if exceeded != "spin" else 2.1
    _, failed, _, _ = env.outcomes()
    assert failed.item() == (exceeded is not None)


@pytest.mark.parametrize("task", ["sticking", "sliding", "tipping", "twisting"])
def test_first_event_survives_rebound_but_not_prior_failure(task):
    env, _, _ = tensor_env(task, n=3)
    counter = [0]

    def step():
        counter[0] += 1

    contact = env.zeros(6)
    contact[:, 2] = 9.81
    env.last_gap = env.zeros()
    env.sim = SimpleNamespace(
        data=SimpleNamespace(qfrc_applied=env.zeros(6), qfrc_constraint=contact), step=step
    )

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
    torch.testing.assert_close(
        duration, torch.tensor([4.0, 2.0, float(env.substeps)]) * env.cfg.physics_dt
    )
    torch.testing.assert_close(error, torch.tensor([4.0, 2.0, float(env.substeps)]))


@pytest.mark.parametrize("task,angle", [("tipping", 80.0), ("twisting", 45.0)])
def test_target_is_first_passage_without_new_face_contact(task, angle):
    env, c, rot = tensor_env(task)
    theta = torch.deg2rad(torch.tensor(angle + 0.01))
    q = torch.zeros(1, 4)
    q[:, 0] = torch.cos(theta / 2)
    q[:, 2 if task == "tipping" else 3] = torch.sin(theta / 2)
    rot[:] = rotation(q)
    if task == "tipping":
        edge_local = torch.tensor([[0.1, 0, -0.1]])
        c[:] = torch.tensor([[0.1, 0, 0]]) - (rot @ edge_local[..., None]).squeeze(-1)
    success, failed, _, _ = env.outcomes()
    assert success.item() and not failed.item()


def test_overhanging_face_and_low_load_are_not_success():
    env, c, _ = tensor_env("sliding")
    env.origin[:, 0] = 0.15
    c[:, 0] = 0.35  # Center is on tray, but a supporting vertex is outside.
    assert env.outcomes()[1].item()
    c[:, 0] = 0
    env.normal_fraction[:] = 0.05
    assert env.outcomes()[1].item()
