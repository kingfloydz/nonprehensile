import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from nonprehensile import pipeline


@pytest.mark.parametrize("gl_backend", ["disable", "egl"])
def test_selection_uses_validation_and_keeps_one_seed(tmp_path, monkeypatch, gl_backend):
    calls = []
    monkeypatch.setenv("MUJOCO_GL", gl_backend)
    monkeypatch.setattr(pipeline, "build", lambda *a, **k: None)

    def process(command, **kwargs):
        assert kwargs["env"]["MUJOCO_GL"] == gl_backend
        assert command[0] == sys.executable
        assert Path(command[1]).name == "cli.py"
        kind = command[2]
        output = Path(command[command.index("--output") + 1])
        if kind == "train":
            config = json.loads(Path(command[command.index("--config") + 1]).read_text())
            env = config["env"]
            # The middle weights win, independently of shaped return.
            score = (
                sum(
                    env[key] == value
                    for key, value in (
                        ("distance_weight", 1.0),
                        ("margin_weight", 0.1),
                        ("startup_weight", 1.0),
                    )
                )
                / 3
            )
            metrics = dict(
                episodes=2,
                success_rate=score,
                success_time=1.0,
                penalized_time=5.0 - 4.0 * score,
                physical_failure_rate=1.0 - score,
                disturbance=0.0,
            )
            (output / "validation.json").write_text(json.dumps(metrics))
            calls.append(config)
        else:
            output.write_text(
                json.dumps(
                    [
                        dict(
                            episodes=2,
                            success_rate=1.0,
                            success_time=1.0,
                            penalized_time=1.0,
                            physical_failure_rate=0.0,
                            disturbance=0.0,
                        )
                    ]
                )
            )

    monkeypatch.setattr(pipeline.subprocess, "run", process)
    pipeline.run(
        tmp_path,
        [0, 1],
        num_envs=2,
        search_iterations=2,
        final_iterations=5,
        bank_size=4,
        workers=1,
    )
    assert len(calls) == 164
    winners = list((tmp_path / "best").glob("*/*/config.json"))
    assert len(winners) == 19
    for path in winners:
        config = json.loads(path.read_text())
        assert config["env"]["seed"] == 1
        assert config["ppo"]["iterations"] == 5
        assert config["env"]["distance_weight"] == 1.0
        assert config["env"]["margin_weight"] == 0.1
        assert config["env"]["startup_weight"] == 1.0
    assert (tmp_path / "results.csv").exists()


def test_headless_default_before_mujoco_import():
    env = dict(os.environ)
    env.pop("MUJOCO_GL", None)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import nonprehensile, os; "
            "assert os.environ['MUJOCO_GL'] == 'disable'; "
            "import mujoco, sys; "
            "assert 'mujoco.egl' not in sys.modules; "
            "mujoco.MjModel.from_xml_string('<mujoco/>')",
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
