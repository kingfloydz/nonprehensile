"""F stage two: deterministic teacher labels, then student-rollout DAgger."""

import json
from dataclasses import asdict, dataclass, replace

import torch

from .env import EnvConfig, TrayEnv
from .ppo import Student, execute_action, save_policy


@dataclass
class DistillConfig:
    samples: int = 1_000_000
    epochs: int = 10
    batch_size: int = 4096
    learning_rate: float = 3e-4
    dagger_rounds: int = 10
    dagger_samples: int = 100_000
    dagger_epochs: int = 3


@torch.no_grad()
def collect(teacher, student, env, samples):
    """Collect on GPU; student=None gives the initial teacher-rollout dataset."""
    steps = (samples + env.n - 1) // env.n
    observations = env.zeros(36).new_empty(steps, env.n, 36)
    targets = observations.new_empty(steps, env.n, 6)
    valid = torch.empty((steps, env.n), dtype=torch.bool, device=env.device)
    obs = env.observe()
    for t in range(steps):
        mean, _ = teacher(obs)
        label, invalid = execute_action(teacher, env, mean)
        observations[t], targets[t], valid[t] = obs, label, ~invalid
        action = label if student is None else student.actor(obs)
        obs, _, _, _ = env.step(action, invalid)
    obs, target, keep = (x.flatten(0, 1)[:samples] for x in (observations, targets, valid))
    # One compaction per collection, not host-side filtering on every control step.
    return obs[keep], target[keep], 1 - keep.float().mean().item()


def fit(student, optimizer, observations, targets, epochs, batch_size):
    if not len(observations):
        raise RuntimeError("No valid teacher labels: inspect teacher mapping failures.")
    loss_sum = observations.new_zeros(())
    for _ in range(epochs):
        for batch in torch.randperm(len(observations), device=observations.device).split(
            batch_size
        ):
            loss = (student.actor(observations[batch]) - targets[batch]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach() * len(batch)
    return (loss_sum / (epochs * len(observations))).item()


def distill(teacher, config, bank, output, device):
    cfg = DistillConfig(**config.get("distill", {}))
    if min(cfg.samples, cfg.epochs, cfg.batch_size, cfg.dagger_samples, cfg.dagger_epochs) <= 0:
        raise ValueError("Distillation sample counts, epochs and batch size must be positive.")
    if cfg.dagger_rounds < 0:
        raise ValueError("dagger_rounds must be nonnegative.")
    (output / "distill_config.json").write_text(json.dumps(asdict(cfg), indent=2))
    env_cfg = replace(EnvConfig(**config["env"]), group="A", disturbance=0)
    env = TrayEnv(env_cfg, bank, device)
    student = Student(36).to(device)
    teacher.eval()
    optimizer = torch.optim.Adam(student.parameters(), lr=cfg.learning_rate)
    observations, targets, rejected = collect(teacher, None, env, cfg.samples)
    with (output / "distillation.jsonl").open("w") as stream:
        for round_ in range(cfg.dagger_rounds + 1):
            if round_:
                env.reset(env.all_ids)
                obs, labels, rejected = collect(teacher, student, env, cfg.dagger_samples)
                observations = torch.cat((observations, obs))
                targets = torch.cat((targets, labels))
            mse = fit(
                student,
                optimizer,
                observations,
                targets,
                cfg.dagger_epochs if round_ else cfg.epochs,
                cfg.batch_size,
            )
            row = dict(
                round=round_, samples=len(observations), mse=mse, rejected_label_fraction=rejected
            )
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            print(json.dumps(row), flush=True)
    save_policy(student, 36, config, output / "policy.pt")
    # Deployment is just an MLP: no geometry, critic, simulator or teacher needed.
    torch.jit.trace(student.actor.eval(), env.observe()[:1]).save(str(output / "student_actor.pt"))
    return student.eval()
