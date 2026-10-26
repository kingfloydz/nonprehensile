# Validation of four-mode-v2

These checks establish that the selected benchmark admits useful reference
motions and that the training implementation runs. They do **not** establish
trained A–F success rates, a continuous-domain feasibility proof, or G1 limits.

## Native MuJoCo reference trajectories

Files in `reference/` retain parameter samples and per-trajectory results.
Mass is 1 kg. Control period is 5 ms. Each case was run at both 0.5 ms and
0.25 ms physics steps. The three non-twisting domains use 64 independent LHS
samples (seed 77121); the final twisting domain uses 64 new LHS samples (79121).
All combinations of parameter endpoints were also checked.

| Mode | Random samples | Endpoint combinations | Total runs, both steps | Passed |
|---|---:|---:|---:|---:|
| sticking | 64 | 16 | 160 | 160 |
| sliding | 64 | 64 | 256 | 256 |
| tipping | 64 | 16 | 160 | 160 |
| twisting | 64 | 64 | 256 | 256 |

The final twisting checks specifically use **45 degrees, mu in [0.05,0.12],
vertical acceleration within ±2 m/s²**, angular speed ≤8 rad/s and yaw acceleration
within ±30 rad/s². The reference pulse plans one degree past the success threshold
to avoid a numerically fragile exact turning-point equality. For mu>0.10 it uses
d=9.81*(1-0.10/mu), az=-d, with horizontal compensation of Euler/centripetal terms.
This is a diagnostic reference only; it is **not** part of any A–F policy/controller.

Across the final native twisting checks, maximum tray speed was 0.691 m/s,
maximum angular speed 7.770 rad/s, maximum downward travel 14.5 cm and terminal
downward speed 0.689 m/s. No subsequent vertical braking was simulated. Earlier
mu≤0.15 tests with ±4 m/s² are not evidence for this final configuration and are
not included here.

The reference acceptance checks used ≤1 mm mode position error, ≤1 degree
unwanted rotation, ≤0.5 mm gap, normal load averaged over 20 ms >0.1mg, and
speed/action/support limits. Sticking used an even tighter 2 mm endpoint threshold
than the training environment's 5 mm. Reference motion completion for the other
three modes is first passage, not terminal rest. Small ideal-equality residuals
can coexist with passing physical tolerances; these are not exact DD certificates.

## Actual training-backend integration

We additionally executed full reference trajectories through `TrayEnv` itself,
using mjlab / MuJoCo-Warp on CPU, the same reset, fictitious forces, observations,
action scaling and substep termination code used by training.

Initial comparison: 16 sampled episodes per mode, at 0.5 and 0.25 ms. Sliding
passed 14/16 at 0.5 ms and 16/16 at 0.25 ms; all other cells passed 16/16.
The two sliding failures involved contact chatter and strict load/gap limits.
Raw results, including failures, are retained in `reference/timestep_comparison.json`.
Consequently the **default training physics step is 0.25 ms**. We did not relax
the physical tolerances, reduce the friction range or discard failing samples.

After selecting that step, all **64 distinct bank rows per mode** (bank seed
79122) were checked at 0.25 ms: **256/256 passed**. See
`reference/training_backend_results.json`. Reproduce this integration check with:

```bash
python tools/check_reference.py --output runs/reference-check --count 64
```

Use `--device cuda:0` for server verification. `--physics-dt 0.0005 0.00025`
compares steps. The tool generates its own diagnostic bank, runs references, saves
per-instance results and exits unsuccessfully if any reference fails. It is not
called by the training pipeline and supplies no privileged controller to E/F.

## Code checks

54 tests passed, including independent LP comparisons for DD/scalar elimination,
action-scale consistency, target/event tests, fixed mass and randomized inertia,
legacy-bank rejection, paired friction-bin evaluation and actual
PPO → BC → DAgger → TorchScript student export. Ruff checks and formatting passed.

Local versions: mjlab 1.6.0, MuJoCo and MuJoCo-Warp 3.11.0, PyTorch 2.13.0,
NumPy 2.5.2, SciPy 1.18.0. No NVIDIA driver was available locally, so GPU/H200
performance and numerical parity remain to be measured on the training server.
The Windows installation contains an unrelated incompatible TensorFlow package;
tests selected TensorBoard's supported no-TensorFlow import path in-process:

```python
import sys, types, pytest
sys.modules['tensorboard.compat.notf'] = types.ModuleType('tensorboard.compat.notf')
raise SystemExit(pytest.main(['tests', '-q', '--basetemp=<fresh-writable-path>']))
```

This changes only optional TensorBoard loading; simulation, PPO, geometry and
distillation ran without mocks in their integration tests. No installed packages
were changed. On a normal PyTorch-only installation, `python -m pytest tests -q`
is sufficient.
