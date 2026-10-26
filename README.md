# Nonprehensile tray RL

Four independent tasks compare A–E reward ablations and a constrained teacher /
distilled student F. Every method observes the same state and commands **all six
tray acceleration components**. Built on mjlab 1.6.0 / MuJoCo-Warp and PyTorch PPO
with the official RSL-RL MLP. This is a tray-object benchmark, not a G1 hardware model.

## Run

Use Linux, Python 3.12, GPU-enabled mjlab and a compatible NVIDIA driver:

```bash
git clone https://github.com/kingfloydz/nonprehensile.git
cd nonprehensile
python3 -m pip install -e .
python3 src/nonprehensile/pipeline.py --gpus 0 --num-envs 4096 --root runs/four-mode-v2-seed1 --seed 1
```

Repeat with seeds 2 and 3 and distinct output roots. Bank seeds stay 101/202/303
for train/validation/test: all methods and training seeds share physical samples.
Multiple GPUs schedule independent runs (`--gpus 0 1 2 3`). `--groups A F` runs
just those methods. Subprocesses use the same interpreter and inherit MUJOCO_GL.

Each seed runs 26 weight-search candidates and 23 final runs. Sticking E equals D
and is omitted. Validation selects weights by success rate, then penalized time,
never shaped return. F trains its own teacher then BC/DAgger; no baseline policy
initializes it. Test data never select settings. Old runs are not modified.

**Rebuild banks and retrain old checkpoints.** The `four-mode-v2` version tag
rejects legacy banks/checkpoints, including 36-input policies with old action
semantics. Use a fresh output root.

## Shared experimental settings

`src/nonprehensile/settings.py` defines all four domains. Dimensions are full box
lengths; ranges are independent uniforms. Mass is **1 kg**; uniform-box inertia
is recomputed from each sampled size. Tray: 80 x 80 cm. Static and kinetic
friction share the sampled coefficient.

| Mode | Goal | Length / width / height (cm) | Friction |
|---|---|---|---|
| sticking | transport 50 cm along world x and stop | 15–25 / 15–25 / 10–20 | 0.20–0.50 |
| sliding | slide 20 cm along tray x | 15–25 / 10–20 / 5–8 | 0.15–0.30 |
| tipping | rotate 80 degrees about the fixed +x bottom edge | 10–12 / 10–15 / 25–30 | 0.60–0.80 |
| twisting | rotate 45 degrees about the normal axis through COM | 25–30 / 25–30 / 5–8 | 0.05–0.12 |

Sliding/tipping start at x=-10 cm. Sliding initial tray roll/pitch each range over
±2 degrees. Twisting initial horizontal COM placement relative to the tray origin
ranges over ±5 mm per axis (not an internal COM offset).

| Mode | Linear speed norm (m/s) | Angular speed norm (rad/s) | Linear acceleration bounds (m/s²) | Angular acceleration bounds (rad/s²) |
|---|---:|---:|---|---|
| sticking | 1.5 | 2 | ±(2,1,1) | ±(2,2,2) |
| sliding | 3 | 2 | ±(5,1,1) | ±(2,2,2) |
| tipping | 3 | 2 | ±(6,1,1) | ±(2,2,2) |
| twisting | 1.5 | 8 | ±(2,2,2) | ±(2,2,30) |

There is **no separate downward-acceleration experiment or scripted unloading
controller**. Twisting policies can translate and rotate together. True friction
remains in the generators; acceleration affects the required wrench.

Control / physics periods: **5 ms / 0.25 ms**; timeout: 5 s. The smaller physics
step avoids contact chatter observed in sliding at 0.5 ms in the training backend.
PPO uses 128-step rollouts (0.64 s), gamma=0.99**0.25 and
GAE lambda=0.95**0.25 to preserve physical discount/rollout times. Each iteration
contains 128 * num_envs transitions, four times the former transition count.

Banks store task-specific geometry, friction, DD constraints and an 80 ms static
settling pose. Mass/inertia, placement and tilt are set before settling using the
shared contact XML. Initial velocities are zero. Settling is offline; resetting
one world does not advance others. Resets sample finite bank rows, not fresh
continuous parameters. Increase --bank-size for greater empirical coverage.

## Task completion and contact checks

All modes latch the first success/failure at physics substeps; failure takes
precedence. Diagnostics stop at that event even if the batched simulator finishes
the control step. Position error must remain ≤1 mm and unwanted rotation ≤1°:

- sticking: relative horizontal COM displacement and relative rotation;
- sliding: transverse displacement and relative rotation;
- tipping: displacement of both fixed-edge endpoints and off-axis rotation;
- twisting: horizontal COM displacement and tilt.

Minimum bottom-corner gap must be ≤0.5 mm and the 20 ms average normal load ≥0.1mg.
The load window is initialized from the settled static load. Any substep without
normal load is separately logged: positive average load alone does not establish
uninterrupted contact. Full-face modes check all four support vertices against
the tray; tipping checks its two supporting edge vertices.

Sticking endpoint error must be <5 mm, speed <0.03 m/s and spin <0.1 rad/s.
Sliding/twisting end on first target passage; tipping ends at 80°, before new-face
impact. These three tasks **do not require terminal rest or recovery to sticking**.
Reverse-speed thresholds are -0.03 m/s (sliding) and -0.10 rad/s (rotating modes).

Speed limits are termination checks, not additional contact-polyhedron rows.
The instantaneous teacher does not certify trajectory-level speed safety.
There is no height constraint or terminal vertical braking phase. Downward travel
and terminal vertical velocity are reported; angle completion is not a complete
stop-and-recover maneuver.

## Methods

All groups share -dt +10*success -10*failure (including timeout) and potential
shaping gamma*Phi(next)-Phi(now), with zero terminal potential. All progress and
observation terms use configured targets. Extra costs are multiplied by dt.

| Group | Additional mechanism |
|---|---|
| A | task reward only |
| B | unwanted-motion penalty |
| C | squared distance to acceleration set after entry |
| D | C plus relative-margin penalty |
| E | D plus near-rest startup-branch cost until measured entry |
| F | constrain teacher actions, then distill an unconstrained student |

Margin penalty is softplus((0.05-relative_margin)/0.05), positive even inside the
set and decreasing with margin. Relative margin is within the affine hull, not a
positive ambient 6D ball for sliding/twisting. Distance includes equality errors.
Fixed perturbation units are (30,30,30,15,15,15) for [alpha,a], independent of
method/action limits. Distance uses 48 FISTA iterations and is approximate.

Entry is latched from measured speed >0.02 m/s (sliding) or >0.10 rad/s (rotating).
E zeros relative velocities in its pre-entry approximation; this is not a full
hybrid-transition certificate. The common bound beta >= -v_parallel/(2*dt)
permits braking, uses measured speed and allows beta=0 at rest. The shared goal
reward must encourage startup. Weight grids remain in pipeline.grid.

## Geometry and F

The offline builder intersects the generator cone with f_z=1, computes facets
with Qhull in its affine hull, then homogenizes and retains equalities.
Sticking/tipping use an inscribed eight-direction friction polygon;
sliding/twisting use fixed Coulomb rays. Twisting uses a four-corner pressure
model, not a continuous pressure distribution. Online Newton–Euler pullback and
scalar elimination give acceleration constraints. Per-task action bounds are
shared by the simulator, margin calculation and teacher mapping.

For Az<=b, Ez=e and -1<=z<=1, F finds a verified relative-interior point z0 and
nullspace basis Q. With G=AQ and h=b-Az0 its map is
z=z0+Q*u/(1+max_i(G_i*u/h_i)). PPO scores latent Gaussian u, not mapped actions.
Latent dimensions are 6/4/6/4. Final residual checks reject unverified actions;
no zero-action fallback is claimed feasible. A failed interior search is not
proof of emptiness. The six-output student uses only an MLP at deployment and
inherits no hard feasibility guarantee. Teacher and student are scored separately.

The payload is independently simulated in the accelerating tray frame. Per-substep
fictitious force/torque includes centrifugal, Coriolis and gyroscopic terms;
tray world pose is integrated separately. There is no contact enforcement or
heavy-tray approximation. All online physics/geometry uses the selected device.

## Observations, outputs and evaluation

Shared 36 inputs: COM position /0.4 (3), rotation (9), relative velocity /2 (3),
relative angular velocity /3 (3), tray velocity in tray frame /2 (3), tray angular
velocity /3 (3), gravity /9.81 (3), task-normalized dimensions (3), task-normalized
friction (1), normalized goal error (3), entry flag (1), time/timeout (1).
Every method receives exact geometry/friction; this is not a perception test.

Evaluation uses paired first-episode batches, not repeated episodes favoring fast
failures. Noise levels 0/.01/.03/.05 refer to fixed metric units before action
clipping. Nominal constraint violations use normalized rows and tolerance 1e-4
before disturbance/reset. JSON results include success/time, physical/mapping
failures, timeouts, substep maximum mode error/gap, normal load, no-contact rate,
downward travel and terminal vertical velocity. Five fixed friction bins report
sample counts and success rates, averaging over the shared size distribution.

Main figures: **learning curves and success versus friction for each mode**, plus
a representative trajectory. Compare training seeds; do not infer RL superiority
from reference-trajectory success. Sticking E and D are the same method here.

Outputs: search/<task>/<group>/<candidate>, best/<task>/<group>, results.csv.
F retains teacher.pt, student policy.pt, student_actor.pt, separate test results,
and distillation logs. Search and final budgets default to 300/2000 iterations;
validation/test sizes default to 2048/8192 (F final validation: 4096).

## Validation

```bash
python -m pip install pytest
python -m pytest tests -q
```

Tests cover DD/scalar elimination against independent LPs, teacher mapping,
physical-unit scaling, targets, event ordering, physical contact checks,
MuJoCo-Warp stepping, and a small PPO→BC→DAgger→student export run.
See docs/validation.md for reference-trajectory evidence and limitations.
H200 throughput, trained success rates and G1 execution require separate tests.
