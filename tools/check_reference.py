"""Diagnostic trajectories in the actual training environment, not RL policies.

Run from a checkout with `python tools/check_reference.py --output <new-dir>`.
No references are used by A-F training. This checks settings/physics integration,
not learning performance or continuous-domain feasibility.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nonprehensile.bank import build  # noqa: E402
from nonprehensile.env import EnvConfig, TrayEnv  # noqa: E402
from nonprehensile.settings import TASKS  # noqa: E402


def reference(env, time):
    c, _, _, wr, gravity, _ = env.state()
    action = env.zeros(6)
    if env.cfg.task == "sticking":
        cap = 0.8 * torch.minimum(env.mu * 9.81, env.size[:, 0] / env.size[:, 2] * 9.81)
        cap = torch.minimum(cap, 0.8 * env.scale[3])
        phase = torch.ceil(torch.sqrt(env.cfg.goal / cap) / env.dt) * env.dt
        acceleration = env.cfg.goal / phase.square()
        action[:, 3] = torch.where(
            time < phase - 1e-7,
            acceleration,
            torch.where(time < 2 * phase - 1e-7, -acceleration, 0),
        )
    elif env.cfg.task == "sliding":
        action[:, 3] = gravity[:, 0] - env.mu * (-gravity[:, 2]) - 1
        action[:, 4] = gravity[:, 1]
    elif env.cfg.task == "twisting":
        down = (9.81 * (1 - 0.1 / env.mu)).clamp_min(0)
        friction = env.mu * env.mass * (-gravity[:, 2] - down)
        friction *= (env.size[:, :2] / 2).norm(dim=1) / env.diagonal[:, 2]
        maximum = env.scale[2]
        planned = env.cfg.goal + torch.pi / 180
        phase = (
            torch.ceil(
                torch.sqrt(2 * planned * friction / (maximum * (maximum - friction))) / env.dt
            )
            * env.dt
        )
        acceleration = (
            friction + torch.sqrt(friction.square() + 8 * planned * friction / phase.square())
        ) / 2
        action[:, 2] = torch.where(time < phase - 1e-7, -acceleration, 0)
        action[:, 5] = -down
        horizontal = gravity - torch.linalg.cross(action[:, :3], c)
        horizontal -= torch.linalg.cross(env.omega, torch.linalg.cross(env.omega, c))
        action[:, 3:5] = horizontal[:, :2]
    else:
        h, _ = env.constraints()
        coefficient, bias = h[:, :, 3], h[:, :, 6]
        ratio = -bias / torch.where(coefficient.abs() > 1e-10, coefficient, 1)
        low = torch.where(coefficient < -1e-10, ratio, -torch.inf).amax(1)
        high = torch.where(coefficient > 1e-10, ratio, torch.inf).amin(1)
        low, high = low.clamp_min(-env.scale[3]), high.clamp_max(env.scale[3])
        beta_ax = -h[:, -1, 3]
        beta_bias = -h[:, -1, 6] - wr[:, 1] / (2 * env.dt)
        desired = (4 - beta_bias) / beta_ax
        action[:, 3] = torch.maximum(
            low + 0.05 * (high - low), torch.minimum(desired, high - 0.05 * (high - low))
        )
        impossible = ((coefficient.abs() <= 1e-10) & (bias > 1e-7)).any(1) | (high <= low)
        action[impossible] = torch.nan
    invalid = ~torch.isfinite(action).all(1) | (action.abs() > env.scale + 1e-5).any(1)
    return action / env.scale, invalid


def check(task, bank, count, dt, device):
    env = TrayEnv(EnvConfig(task=task, num_envs=count, physics_dt=dt, seed=79123), bank, device)
    env.reset(env.all_ids, bank_indices=torch.arange(count, device=device))
    alive = torch.ones(count, dtype=torch.bool, device=device)
    success = torch.zeros_like(alive)
    result = {
        k: env.zeros()
        for k in (
            "time",
            "mode_error",
            "contact_gap",
            "tray_drop",
            "terminal_vertical_velocity",
            "min_normal_load",
        )
    }
    result["min_normal_load"][:] = 1
    ids = env.bank_ids.cpu().tolist()
    sizes, mus = env.size.cpu().tolist(), env.mu.cpu().tolist()
    for step in range(env.max_steps):
        action, invalid = reference(env, step * env.dt)
        _, _, done, info = env.step(action, invalid)
        for key in ("mode_error", "contact_gap", "tray_drop"):
            result[key] = torch.where(alive, torch.maximum(result[key], info[key]), result[key])
        result["min_normal_load"] = torch.where(
            alive,
            torch.minimum(result["min_normal_load"], info["min_normal_load"]),
            result["min_normal_load"],
        )
        first = alive & done
        success[first] = info["success"][first]
        for key in ("time", "terminal_vertical_velocity"):
            result[key][first] = info[key][first]
        alive &= ~done
        if not alive.any():
            break
    records = [
        dict(
            task=task,
            physics_dt=dt,
            bank_id=ids[i],
            size=sizes[i],
            mu=mus[i],
            success=bool(success[i]),
            **{k: float(v[i]) for k, v in result.items()},
        )
        for i in range(count)
    ]
    print(task, dt, int(success.sum()), "/", count, flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--physics-dt", type=float, nargs="+", default=[0.00025])
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    bank = args.output / "bank.npz"
    build(bank, count=max(64, args.count), seed=79122, workers=1)
    rows = []
    for task in TASKS:
        for dt in args.physics_dt:
            rows.extend(check(task, bank, args.count, dt, args.device))
            (args.output / "reference_results.json").write_text(json.dumps(rows, indent=2))
    if not all(row["success"] for row in rows):
        raise SystemExit("Some reference trajectories failed; inspect reference_results.json")


if __name__ == "__main__":
    main()
