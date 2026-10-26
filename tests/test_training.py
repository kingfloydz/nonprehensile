import json

import torch

from nonprehensile import ppo
from nonprehensile.bank import build
from nonprehensile.pipeline import configuration
from nonprehensile.env import EnvConfig


def test_teacher_ppo_distillation_and_deployment(tmp_path, monkeypatch):
    """Real physics/PPO/BC/DAgger; shorten only the 250-step validation loop."""
    bank = tmp_path / "bank.npz"
    build(bank, count=4, workers=1)
    config = configuration("sliding", "F", 2, 1)
    config["ppo"].update(
        rollout_steps=2,
        epochs=1,
        minibatches=1,
        compile=False,
        log_interval=1,
        validation_episodes=2,
    )
    config["distill"] = dict(
        samples=4, epochs=1, batch_size=4, dagger_rounds=1, dagger_samples=4, dagger_epochs=1
    )
    kinds = []

    def brief_evaluate(policy, cfg, bank, episodes, device, noise=0):
        kinds.append(policy.kind)
        assert policy.actions == (4 if policy.kind == "teacher" else 6)
        obs = torch.zeros(2, 36)
        mean, _ = policy(obs)
        assert torch.isfinite(mean).all()
        return dict(episodes=episodes, success_rate=0.0, disturbance=noise)

    monkeypatch.setattr(ppo, "evaluate", brief_evaluate)
    ppo.train(config, bank, bank, tmp_path, device="cpu")
    assert kinds == ["teacher", "student"]
    teacher = torch.load(tmp_path / "teacher.pt", weights_only=False)
    student = torch.load(tmp_path / "policy.pt", weights_only=False)
    assert teacher["actions"] == 4 and teacher["kind"] == "teacher"
    assert student["actions"] == 6 and student["kind"] == "student"
    assert all(key.startswith("actor.") for key in student["policy"])
    deployed = torch.jit.load(tmp_path / "student_actor.pt")
    model = ppo.Student(36)
    model.load_state_dict(student["policy"])
    obs = torch.randn(7, 36)
    torch.testing.assert_close(deployed(obs), model.actor(obs))
    assert deployed(obs).abs().max() <= 1
    for name in ("teacher", "policy"):
        ppo.test(tmp_path / f"{name}.pt", bank, tmp_path / f"{name}_test.json", 2, "cpu")
    rows = [json.loads(row) for row in (tmp_path / "distillation.jsonl").read_text().splitlines()]
    assert [r["round"] for r in rows] == [0, 1]
    assert rows[-1]["samples"] >= rows[0]["samples"] > 0


def test_student_action_does_not_query_geometry():
    class Environment:
        entered = torch.zeros(3, dtype=torch.bool)

        def constraints(self):
            raise AssertionError("Deployment must not call the teacher mapping")

    student = ppo.Student(36)
    raw, _ = student(torch.randn(3, 36))
    action, invalid = ppo.execute_action(student, Environment(), raw)
    torch.testing.assert_close(action, raw)
    assert not invalid.any()


def test_actual_evaluation_reports_paired_friction_bins_and_timeout(tmp_path):
    bank = tmp_path / "bank.npz"
    build(bank, count=4, workers=1)
    student = ppo.Student(36)
    with torch.no_grad():
        for parameter in student.parameters():
            parameter.zero_()
    cfg = EnvConfig(task="twisting", num_envs=2, timeout_s=0.02)
    first = ppo.evaluate(student, cfg, bank, 2, "cpu")
    second = ppo.evaluate(student, cfg, bank, 2, "cpu")
    assert first == second
    assert first["success_rate"] == 0 and first["timeout_rate"] == 1
    assert sum(item["episodes"] for item in first["friction_bins"]) == 2
    assert first["mean_abs_terminal_vertical_speed"] == 0
    assert first["mean_episode_min_normal_load"] > 0.1
