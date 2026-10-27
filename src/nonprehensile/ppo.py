"""Task adapters for the official RSL-RL PPO used by mjlab."""

import json
import math
import time
from dataclasses import replace
from pathlib import Path

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.modules import MLP
from tensordict import TensorDict
from torch import nn

from .env import EnvConfig, TrayEnv
from .mapping import GaugeMap, normalized_constraints, violation
from .settings import BENCHMARK_VERSION, SETTINGS


def observations_dict(obs):
    return TensorDict({"state": obs}, batch_size=[obs.shape[0]])


class Policy(nn.Module):
    """Checkpoint/evaluation facade; distributions and models belong to RSL-RL."""

    def __init__(self, observations, hidden=256, actions=6, kind="direct"):
        super().__init__()
        self.kind, self.actions = kind, actions
        obs = observations_dict(torch.zeros(1, observations))
        groups = {"actor": ["state"], "critic": ["state"]}
        self.actor = MLPModel(
            obs,
            groups,
            "actor",
            actions,
            hidden_dims=[hidden, hidden],
            activation="tanh",
            distribution_cfg=dict(
                class_name="GaussianDistribution",
                std_type="log",
                init_std=math.exp(-0.7),
                std_range=(math.exp(-3), math.e),
            ),
        )
        self.critic = MLPModel(
            obs, groups, "critic", 1, hidden_dims=[hidden, hidden], activation="tanh"
        )

    def forward(self, obs):
        obs = observations_dict(obs)
        return self.actor(obs), self.critic(obs).squeeze(-1)


def require_finite(**tensors):
    for name, tensor in tensors.items():
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(f"Non-finite {name}")


def check_optimizer_step(optimizer, args, kwargs):
    """Abort before Adam writes non-finite gradients into otherwise valid weights."""
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if parameter.grad is not None:
                require_finite(gradient=parameter.grad)


class Student(nn.Module):
    kind, actions = "student", 6

    def __init__(self, observations):
        super().__init__()
        self.actor = MLP(observations, 6, [256, 256], activation="tanh", last_activation="tanh")

    def forward(self, obs):
        return self.actor(obs), obs.new_zeros(obs.shape[0])


def execute_action(policy, env, raw):
    if policy.kind == "teacher":
        return GaugeMap(
            *env.constraints(), iterations=env.cfg.mapping_iterations, action_scale=env.scale
        )(raw)
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
    mode_errors, contact_gaps = [], []
    frictions, drops, vertical_speeds, timeouts, load_minima, no_contacts = [], [], [], [], [], []
    violations, transitions = torch.zeros((), device=device), torch.zeros((), device=device)
    with torch.inference_mode():
        for batch in range((episodes + env.n - 1) // env.n):
            env.rng.manual_seed(10000 + batch)
            env.noise_rng.manual_seed(20000 + batch)
            env.reset(env.all_ids)
            frictions.append(env.mu.cpu().clone())
            obs = env.observe()
            alive = torch.arange(env.n, device=device) < min(env.n, episodes - batch * env.n)
            success, duration, failure, mapping_failure = [env.zeros() for _ in range(4)]
            max_error, max_gap = env.zeros(), env.zeros()
            max_drop, final_vz, timed_out = env.zeros(), env.zeros(), env.zeros()
            min_load, lost_contact = torch.ones_like(max_drop), torch.zeros_like(alive)
            for _ in range(env.max_steps):
                mean, _ = policy(obs)
                action, invalid = execute_action(policy, env, mean)
                constraints = normalized_constraints(*env.constraints(), action_scale=env.scale)
                violated = violation(*constraints, action) > 1e-4
                violations += ((violated | invalid) & alive).sum()
                transitions += alive.sum()
                obs, _, done, info = env.step(action, invalid)
                max_error = torch.where(
                    alive, torch.maximum(max_error, info["mode_error"]), max_error
                )
                max_gap = torch.where(alive, torch.maximum(max_gap, info["contact_gap"]), max_gap)
                max_drop = torch.where(alive, torch.maximum(max_drop, info["tray_drop"]), max_drop)
                min_load = torch.where(
                    alive, torch.minimum(min_load, info["min_normal_load"]), min_load
                )
                lost_contact |= alive & info["no_contact"]
                first = alive & done
                success[first] = info["success"][first].float()
                duration[first] = info["time"][first]
                failure[first] = info["physical_failure"][first].float()
                mapping_failure[first] = info["mapping_failure"][first].float()
                final_vz[first] = info["terminal_vertical_velocity"][first]
                timed_out[first] = info["timeout"][first].float()
                alive &= ~done
                if not alive.any():
                    break
            successes.append(success.cpu())
            times.append(duration.cpu())
            failures.append(failure.cpu())
            mapping_failures.append(mapping_failure.cpu())
            mode_errors.append(max_error.cpu())
            contact_gaps.append(max_gap.cpu())
            drops.append(max_drop.cpu())
            vertical_speeds.append(final_vz.cpu())
            timeouts.append(timed_out.cpu())
            load_minima.append(min_load.cpu())
            no_contacts.append(lost_contact.cpu())
    success = torch.cat(successes)[:episodes].float()
    duration = torch.cat(times)[:episodes]
    fail = torch.cat(failures)[:episodes].float()
    successful = success.sum().item()
    result = dict(
        episodes=episodes,
        success_rate=success.mean().item(),
        success_time=(duration * success).sum().item() / successful if successful else None,
        penalized_time=(duration * success + cfg.timeout_s * (1 - success)).mean().item(),
        physical_failure_rate=fail.mean().item(),
        mapping_failure_rate=torch.cat(mapping_failures)[:episodes].mean().item(),
        constraint_violation_rate=(violations / transitions.clamp_min(1)).item(),
        mean_episode_max_mode_error=torch.cat(mode_errors)[:episodes].mean().item(),
        mean_episode_max_contact_gap=torch.cat(contact_gaps)[:episodes].mean().item(),
        timeout_rate=torch.cat(timeouts)[:episodes].mean().item(),
        mean_episode_max_drop=torch.cat(drops)[:episodes].mean().item(),
        mean_abs_terminal_vertical_speed=torch.cat(vertical_speeds)[:episodes].abs().mean().item(),
        mean_episode_min_normal_load=torch.cat(load_minima)[:episodes].mean().item(),
        any_substep_without_contact_rate=torch.cat(no_contacts)[:episodes].float().mean().item(),
        disturbance=noise,
    )
    mu = torch.cat(frictions)[:episodes]
    edges = torch.linspace(*SETTINGS[cfg.task].friction, 6)
    bins = torch.bucketize(mu, edges[1:-1], right=True)
    result["friction_bins"] = [
        dict(
            low=edges[i].item(),
            high=edges[i + 1].item(),
            episodes=int((bins == i).sum()),
            success_rate=success[bins == i].mean().item() if (bins == i).any() else None,
        )
        for i in range(5)
    ]
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
    steps, n = hp["rollout_steps"], cfg.num_envs
    if steps * n < 2 or not 1 <= hp["minibatches"] <= steps * n:
        raise ValueError("PPO needs at least two transitions and nonempty minibatches")
    storage = RolloutStorage("rl", n, steps, observations_dict(obs), [model.actions], device)
    algorithm = PPO(
        model.actor,
        model.critic,
        storage,
        num_learning_epochs=hp["epochs"],
        num_mini_batches=hp["minibatches"],
        learning_rate=hp["learning_rate"],
        gamma=hp["gamma"],
        lam=hp["gae_lambda"],
        entropy_coef=hp["entropy"],
        value_loss_coef=0.5,
        schedule="adaptive",
        desired_kl=hp.get("desired_kl", 0.01),
        use_clipped_value_loss=True,
        device=device,
    )
    algorithm.optimizer.register_step_pre_hook(check_optimizer_step)
    algorithm.compile("default" if hp["compile"] else None)
    start = time.monotonic()
    with (output / "learning.jsonl").open("w") as stream:
        for iteration in range(hp["iterations"]):
            finished = torch.zeros(4, device=device)
            try:
                with torch.no_grad():
                    for t in range(steps):
                        require_finite(observation=obs)
                        raw = algorithm.act(observations_dict(obs))
                        require_finite(action=raw, value=algorithm.transition.values)
                        action, invalid = execute_action(model, env, raw)
                        obs, reward, done, info = env.step(action, invalid)
                        require_finite(next_observation=obs, reward=reward)
                        # Timeout is task failure here, not an infinite-horizon truncation.
                        # Store the raw latent action, including for the constrained teacher.
                        algorithm.process_env_step(observations_dict(obs), reward, done, {})
                        finished[0] += done.sum()
                        finished[1] += info["success"].sum()
                        finished[2] += (info["time"] * info["success"]).sum()
                        finished[3] += info["mapping_failure"].sum()
                    algorithm.compute_returns(observations_dict(obs))
                    require_finite(advantages=storage.advantages, returns=storage.returns)
                    reward_mean = storage.rewards.mean().item()
                losses = algorithm.update()
                for name, parameter in model.named_parameters():
                    require_finite(**{name: parameter})
            except (FloatingPointError, RuntimeError) as exc:
                torch.save(
                    dict(
                        iteration=iteration + 1,
                        error=str(exc),
                        config=config,
                        algorithm=algorithm.save(),
                        storage=storage,
                        observation=obs,
                        rng=torch.get_rng_state(),
                    ),
                    output / "failure.pt",
                )
                raise RuntimeError(
                    f"PPO failed at iteration {iteration + 1}; diagnostics: {output / 'failure.pt'}"
                ) from exc
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
                    reward=reward_mean,
                    loss=losses["surrogate"]
                    + 0.5 * losses["value"]
                    - hp["entropy"] * losses["entropy"],
                    losses=losses,
                    learning_rate=algorithm.learning_rate,
                )
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                print(json.dumps(row), flush=True)
    save_policy(model, obs.shape[-1], config, output / ("teacher.pt" if teacher else "policy.pt"))
    del algorithm, storage, env
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
    if payload["config"]["env"].get("benchmark_version") != BENCHMARK_VERSION:
        raise ValueError("Old benchmark checkpoint; retrain using four-mode-v2 settings.")
    if payload["observations"] != 36:
        raise ValueError("This version uses 36 observations; retrain older 46-input checkpoints.")
    kind = payload.get("kind", "direct")
    if kind != "student" and "log_std" in payload["policy"]:
        raise ValueError("Legacy custom-PPO checkpoint; retrain with the RSL-RL backend.")
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
