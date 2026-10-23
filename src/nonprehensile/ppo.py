"""Batched tanh-Gaussian PPO, one seed per run."""

import json
import time
from dataclasses import replace
from pathlib import Path

import torch
from rsl_rl.modules import MLP
from torch import nn

from .env import EnvConfig, TrayEnv
from .mapping import GaugeMap, normalized_constraints, violation


class Policy(nn.Module):
    def __init__(self, observations, hidden=256, actions=6, kind="direct"):
        super().__init__()
        self.kind, self.actions = kind, actions
        self.actor = MLP(observations, actions, [hidden, hidden], activation="tanh")
        self.critic = MLP(observations, 1, [hidden, hidden], activation="tanh")
        self.log_std = nn.Parameter(torch.full((actions,), -0.7))

    def forward(self, obs):
        return self.actor(obs), self.critic(obs).squeeze(-1)

    def distribution(self, obs):
        mean, value = self(obs)
        return torch.distributions.Normal(
            mean, self.log_std.clamp(-3, 1).exp(), validate_args=False
        ), value


class Student(nn.Module):
    kind, actions = "student", 6

    def __init__(self, observations):
        super().__init__()
        self.actor = MLP(observations, 6, [256, 256], activation="tanh", last_activation="tanh")

    def forward(self, obs):
        return self.actor(obs), obs.new_zeros(obs.shape[0])


def execute_action(policy, env, raw):
    if policy.kind == "teacher":
        return GaugeMap(*env.constraints(), iterations=env.cfg.mapping_iterations)(raw)
    action = raw if policy.kind == "student" else raw.tanh()
    return action, torch.zeros_like(env.entered)


def save_policy(model, observations, config, path):
    torch.save(
        dict(
            policy=model.state_dict(),
            observations=observations,
            config=config,
            kind=model.kind,
            actions=model.actions,
        ),
        path,
    )


def evaluate(policy, cfg, bank, episodes, device, noise=0.0):
    env = TrayEnv(
        replace(cfg, group="A", num_envs=min(512, episodes), seed=202, disturbance=noise),
        bank,
        device,
    )
    successes, times, failures, mapping_failures = [], [], [], []
    violations, transitions = torch.zeros((), device=device), torch.zeros((), device=device)
    with torch.inference_mode():
        for batch in range((episodes + env.n - 1) // env.n):
            env.rng.manual_seed(10000 + batch)
            env.noise_rng.manual_seed(20000 + batch)
            env.reset(env.all_ids)
            obs = env.observe()
            alive = torch.arange(env.n, device=device) < min(env.n, episodes - batch * env.n)
            success, duration, failure, mapping_failure = [env.zeros() for _ in range(4)]
            for _ in range(250):
                mean, _ = policy(obs)
                action, invalid = execute_action(policy, env, mean)
                constraints = normalized_constraints(*env.constraints())
                violated = violation(*constraints, action) > 1e-4
                violations += ((violated | invalid) & alive).sum()
                transitions += alive.sum()
                obs, _, done, info = env.step(action, invalid)
                first = alive & done
                success[first] = info["success"][first].float()
                duration[first] = info["time"][first]
                failure[first] = info["physical_failure"][first].float()
                mapping_failure[first] = info["mapping_failure"][first].float()
                alive &= ~done
            successes.append(success.cpu())
            times.append(duration.cpu())
            failures.append(failure.cpu())
            mapping_failures.append(mapping_failure.cpu())
    success = torch.cat(successes)[:episodes].float()
    duration = torch.cat(times)[:episodes]
    fail = torch.cat(failures)[:episodes].float()
    successful = success.sum().item()
    result = dict(
        episodes=episodes,
        success_rate=success.mean().item(),
        success_time=(duration * success).sum().item() / successful if successful else None,
        penalized_time=(duration * success + 5 * (1 - success)).mean().item(),
        physical_failure_rate=fail.mean().item(),
        mapping_failure_rate=torch.cat(mapping_failures)[:episodes].mean().item(),
        constraint_violation_rate=(violations / transitions.clamp_min(1)).item(),
        disturbance=noise,
    )
    return result


def train(config, train_bank, validation_bank, output, device="cuda:0"):
    hp = config["ppo"]
    cfg = replace(EnvConfig(**config["env"]), discount=hp["gamma"])
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(4)
    # Geometric equality certificates need full float32 matmul precision.
    torch.backends.cuda.matmul.allow_tf32 = False
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    env = TrayEnv(cfg, train_bank, device)
    obs = env.observe()
    teacher = cfg.group == "F"
    latent = 4 if cfg.task in ("sliding", "twisting") else 6
    model = Policy(
        obs.shape[-1], actions=latent if teacher else 6, kind="teacher" if teacher else "direct"
    ).to(device)
    if hp["compile"]:
        model.forward = torch.compile(model.forward)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["learning_rate"], eps=1e-5)
    steps, n = hp["rollout_steps"], cfg.num_envs
    observations = torch.empty((steps, *obs.shape), device=device)
    actions = torch.empty((steps, n, model.actions), device=device)
    logs, values, rewards, dones = [torch.empty((steps, n), device=device) for _ in range(4)]
    start = time.monotonic()
    with (output / "learning.jsonl").open("w") as stream:
        for iteration in range(hp["iterations"]):
            finished = torch.zeros(4, device=device)
            with torch.no_grad():
                for t in range(steps):
                    distribution, value = model.distribution(obs)
                    raw = distribution.sample()
                    observations[t], actions[t], values[t] = obs, raw, value
                    logs[t] = distribution.log_prob(raw).sum(-1)
                    action, invalid = execute_action(model, env, raw)
                    obs, reward, done, info = env.step(action, invalid)
                    rewards[t], dones[t] = reward, done.float()
                    finished[0] += done.sum()
                    finished[1] += info["success"].sum()
                    finished[2] += (info["time"] * info["success"]).sum()
                    finished[3] += info["mapping_failure"].sum()
                _, next_value = model(obs)
                advantage = torch.empty_like(rewards)
                gae = torch.zeros(n, device=device)
                for t in reversed(range(steps)):
                    continuation = 1 - dones[t]
                    delta = rewards[t] + hp["gamma"] * next_value * continuation - values[t]
                    gae = delta + hp["gamma"] * hp["gae_lambda"] * continuation * gae
                    advantage[t] = gae
                    next_value = values[t]
                returns = advantage + values
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            flat_obs, flat_action = observations.flatten(0, 1), actions.flatten(0, 1)
            flat_log, flat_adv, flat_return = logs.flatten(), advantage.flatten(), returns.flatten()
            for _ in range(hp["epochs"]):
                permutation = torch.randperm(steps * n, device=device)
                for batch in permutation.chunk(hp["minibatches"]):
                    distribution, value = model.distribution(flat_obs[batch])
                    log = distribution.log_prob(flat_action[batch]).sum(-1)
                    ratio = (log - flat_log[batch]).exp()
                    surrogate = torch.minimum(
                        ratio * flat_adv[batch], ratio.clamp(0.8, 1.2) * flat_adv[batch]
                    )
                    loss = -surrogate.mean() + 0.5 * (value - flat_return[batch]).square().mean()
                    loss -= hp["entropy"] * distribution.entropy().sum(-1).mean()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
            if (iteration + 1) % hp["log_interval"] == 0:
                totals = finished.cpu().tolist()
                row = dict(
                    iteration=iteration + 1,
                    transitions=(iteration + 1) * steps * n,
                    elapsed=time.monotonic() - start,
                    completed=totals[0],
                    successes=totals[1],
                    success_time_sum=totals[2],
                    mapping_failures=totals[3],
                    reward=rewards.mean().item(),
                    loss=loss.item(),
                )
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                print(json.dumps(row), flush=True)
    save_policy(model, obs.shape[-1], config, output / ("teacher.pt" if teacher else "policy.pt"))
    del env
    metrics = evaluate(model, cfg, validation_bank, hp["validation_episodes"], device)
    if teacher:
        (output / "teacher_validation.json").write_text(json.dumps(metrics, indent=2))
        from .distill import distill

        model = distill(model, config, train_bank, output, device)
        metrics = evaluate(model, cfg, validation_bank, hp["validation_episodes"], device)
    (output / "validation.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def test(checkpoint, bank, output, episodes=8192, device="cuda:0"):
    torch.set_num_threads(4)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload["observations"] != 36:
        raise ValueError("This version uses 36 observations; retrain older 46-input checkpoints.")
    kind = payload.get("kind", "direct")
    model = (
        Student(payload["observations"])
        if kind == "student"
        else Policy(payload["observations"], actions=payload.get("actions", 6), kind=kind)
    ).to(device)
    model.load_state_dict(payload["policy"])
    model.eval()
    cfg = EnvConfig(**payload["config"]["env"])
    results = [
        evaluate(model, cfg, bank, episodes, device, noise) for noise in (0.0, 0.01, 0.03, 0.05)
    ]
    Path(output).write_text(json.dumps(results, indent=2))
