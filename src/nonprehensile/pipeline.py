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


def grid(group):
    if group == "A":
        return [{}]
    if group == "B":
        return [dict(conventional_weight=w) for w in (0.1, 1.0, 10.0)]
    if group == "C":
        return [dict(distance_weight=w) for w in (0.1, 1.0, 10.0)]
    weights = [
        dict(distance_weight=d, margin_weight=m)
        for d, m in itertools.product((0.1, 1.0, 10.0), (0.01, 0.1, 1.0))
    ]
    if group == "D":
        return weights
    return [dict(w, startup_weight=s) for w in weights for s in (0.3, 1.0, 3.0)]


def run(
    root,
    gpus,
    num_envs=4096,
    search_iterations=300,
    final_iterations=2000,
    bank_size=8192,
    workers=8,
):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    banks = root / "banks"
    for split, seed in (("train", 101), ("validation", 202), ("test", 303)):
        build(banks / f"{split}.npz", bank_size, seed, workers)
    jobs = []
    for task in TASKS:
        for group in "ABCDE":
            if task == "sticking" and group == "E":
                continue
            for index, weights in enumerate(grid(group)):
                directory = root / "search" / task / group / f"{index:03d}"
                directory.mkdir(parents=True, exist_ok=True)
                config = dict(
                    env=asdict(
                        EnvConfig(task=task, group=group, num_envs=num_envs, seed=1, **weights)
                    ),
                    ppo=dict(
                        iterations=search_iterations,
                        rollout_steps=32,
                        epochs=4,
                        minibatches=16,
                        learning_rate=3e-4,
                        gamma=0.99,
                        gae_lambda=0.95,
                        entropy=0.005,
                        compile=True,
                        log_interval=10,
                        validation_episodes=2048,
                    ),
                )
                (directory / "config.json").write_text(json.dumps(config, indent=2))
                jobs.append(directory)
    slots = Queue()
    for gpu in gpus:
        slots.put(str(gpu))

    def execute(directory, final=False):
        gpu = slots.get()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, MUJOCO_GL="egl", OMP_NUM_THREADS="4")
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
                    subprocess.run(
                        [
                            sys.executable,
                            str(Path(__file__).with_name("cli.py")),
                            "test",
                            "--checkpoint",
                            str(directory / "policy.pt"),
                            "--bank",
                            str(banks / "test.npz"),
                            "--output",
                            str(directory / "test.json"),
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
        for group in "ABCDE":
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
                "episodes",
                "success_rate",
                "success_time",
                "penalized_time",
                "physical_failure_rate",
                "disturbance",
            ],
        )
        writer.writeheader()
        for directory in selected:
            for metrics in json.loads((directory / "test.json").read_text()):
                writer.writerow(dict(task=directory.parent.name, group=directory.name, **metrics))
