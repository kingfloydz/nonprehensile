"""One job per GPU; train -> validate -> select -> retrain -> held-out test."""

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from nonprehensile.cli import main

    main(["pipeline", *sys.argv[1:]])
    raise SystemExit

import csv
import itertools
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from queue import Queue

from .bank import TASKS, build
from .env import EnvConfig


def grid(task, group):
    """26 fixed-task candidates; zero-success tasks use diagnostic defaults."""
    if group == "A":
        return [{}]
    if group == "B":
        return [dict(conventional_weight=1.0 if task == "sliding" else 0.1)]
    if group == "C":
        return [dict(distance_weight=10.0 if task == "sliding" else 1.0)]
    if group == "D":
        return [
            dict(distance_weight=10.0 if task == "sliding" else 1.0, margin_weight=m)
            for m in ((0.1, 1.0) if task == "sliding" else (0.1,))
        ]
    if group != "E":
        raise ValueError("Only A-E have reward search grids")
    if task == "sticking":
        return []
    if task == "twisting":
        return [dict(distance_weight=1.0, margin_weight=0.01, startup_weight=0.3)]
    margins, startups = (
        ((0.01, 1.0), (0.3, 3.0)) if task == "sliding" else ((0.01, 0.1), (1.0, 3.0))
    )
    return [
        dict(distance_weight=10.0, margin_weight=m, startup_weight=s)
        for m, s in itertools.product(margins, startups)
    ]


def configuration(task, group, num_envs, iterations, seed=1, **weights):
    return dict(
        env=asdict(EnvConfig(task=task, group=group, num_envs=num_envs, seed=seed, **weights)),
        ppo=dict(
            iterations=iterations,
            rollout_steps=128,
            epochs=4,
            minibatches=16,
            learning_rate=3e-4,
            gamma=0.99**0.25,
            gae_lambda=0.95**0.25,
            entropy=0.005,
            compile=True,
            log_interval=10,
            validation_episodes=2048,
        ),
    )


def run(
    root,
    gpus,
    num_envs=4096,
    search_iterations=300,
    final_iterations=2000,
    bank_size=8192,
    workers=8,
    groups=("A", "B", "C", "D", "E", "F"),
    seed=1,
):
    if not gpus or not groups or any(g not in "ABCDEF" or len(g) != 1 for g in groups):
        raise ValueError("Specify GPUs and groups from A, B, C, D, E, F.")
    root = Path(root).resolve()
    if any((root / name).exists() for name in ("banks", "search", "best", "results.csv")):
        raise FileExistsError("Experiment output already exists; choose a fresh --root.")
    root.mkdir(parents=True, exist_ok=True)
    banks = root / "banks"
    for split, bank_seed in (("train", 101), ("validation", 202), ("test", 303)):
        build(banks / f"{split}.npz", bank_size, bank_seed, workers)
    jobs = []
    for task in TASKS:
        for group in dict.fromkeys(groups):
            if group == "F" or (task == "sticking" and group == "E"):
                continue
            for index, weights in enumerate(grid(task, group)):
                directory = root / "search" / task / group / f"{index:03d}"
                directory.mkdir(parents=True, exist_ok=True)
                config = configuration(
                    task, group, num_envs, search_iterations, seed=seed, **weights
                )
                (directory / "config.json").write_text(json.dumps(config, indent=2))
                jobs.append(directory)
    slots = Queue()
    for gpu in gpus:
        slots.put(str(gpu))

    def execute(directory, final=False):
        gpu = slots.get()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS="4")
        command = [
            sys.executable,
            str(Path(__file__).with_name("cli.py")),
            "train",
            "--config",
            str(directory / "config.json"),
            "--bank",
            str(banks / "train.npz"),
            "--validation-bank",
            str(banks / "validation.npz"),
            "--output",
            str(directory),
        ]
        try:
            with (directory / "console.log").open("w") as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                if final:
                    for model in ("policy", "teacher") if directory.name == "F" else ("policy",):
                        subprocess.run(
                            [
                                sys.executable,
                                str(Path(__file__).with_name("cli.py")),
                                "test",
                                "--checkpoint",
                                str(directory / f"{model}.pt"),
                                "--bank",
                                str(banks / "test.npz"),
                                "--output",
                                str(
                                    directory
                                    / ("test.json" if model == "policy" else "teacher_test.json")
                                ),
                            ],
                            env=env,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            check=True,
                        )
        finally:
            slots.put(gpu)
        return directory

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        list(pool.map(execute, jobs))
    selected = []
    for task in TASKS:
        for group in dict.fromkeys(groups):
            if group == "F":
                from .distill import DistillConfig

                destination = root / "best" / task / group
                destination.mkdir(parents=True, exist_ok=True)
                cfg = configuration(task, group, num_envs, final_iterations, seed=seed)
                cfg["ppo"]["validation_episodes"] = 4096
                cfg["distill"] = asdict(DistillConfig())
                (destination / "config.json").write_text(json.dumps(cfg, indent=2))
                selected.append(destination)
                continue
            candidates = [
                p for p in jobs if p.parent.name == group and p.parent.parent.name == task
            ]
            if not candidates:
                continue

            def score(path):
                m = json.loads((path / "validation.json").read_text())
                return m["success_rate"], -m["penalized_time"]

            winner = max(candidates, key=score)
            destination = root / "best" / task / group
            destination.mkdir(parents=True, exist_ok=True)
            cfg = json.loads((winner / "config.json").read_text())
            cfg["ppo"]["iterations"] = final_iterations
            cfg["ppo"]["validation_episodes"] = 4096
            (destination / "config.json").write_text(json.dumps(cfg, indent=2))
            shutil.copy2(winner / "validation.json", destination / "selection_validation.json")
            winner_score = score(winner)
            selection = dict(
                task_solved=winner_score[0] > 0,
                tied_candidates=sum(score(p) == winner_score for p in candidates),
                rule="success_rate, then penalized_time; first candidate breaks exact ties",
            )
            (destination / "selection.json").write_text(json.dumps(selection, indent=2))
            (destination / "selected_from.txt").write_text(str(winner))
            selected.append(destination)
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        list(pool.map(lambda p: execute(p, True), selected))
    with (root / "results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "task",
                "group",
                "stage",
                "episodes",
                "success_rate",
                "success_time",
                "penalized_time",
                "physical_failure_rate",
                "mapping_failure_rate",
                "constraint_violation_rate",
                "mean_episode_max_mode_error",
                "mean_episode_max_contact_gap",
                "disturbance",
                "timeout_rate",
                "mean_episode_max_drop",
                "mean_abs_terminal_vertical_speed",
                "mean_episode_min_normal_load",
                "any_substep_without_contact_rate",
                "friction_bins",
            ],
        )
        writer.writeheader()
        for directory in selected:
            stages = [("student" if directory.name == "F" else "direct", "test.json")]
            if directory.name == "F":
                stages.append(("teacher", "teacher_test.json"))
            for stage, file in stages:
                for metrics in json.loads((directory / file).read_text()):
                    if "friction_bins" in metrics:
                        metrics["friction_bins"] = json.dumps(metrics["friction_bins"])
                    writer.writerow(
                        dict(
                            task=directory.parent.name, group=directory.name, stage=stage, **metrics
                        )
                    )
