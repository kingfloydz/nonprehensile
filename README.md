# Nonprehensile tray RL

Four independent tasks, five reward ablations (A–E), and a constrained-teacher /
distilled-student method (F), with one training seed (`1`). Built on the official
**mjlab 1.6.0 / MuJoCo-Warp** simulation API and the official RSL-RL `MLP`,
with batched PyTorch PPO.
Each task/group trains a separate policy.
The server pipeline prepares geometry, searches weights, selects on held-out
validation, retrains each selected setting, and exports held-out test results.
There are no automatic retries, CPU fallbacks, or environment-detection scripts.

## H200

Use Linux, Python 3.12, an NVIDIA driver compatible with the CUDA build of PyTorch,
and GPU-enabled mjlab 1.6.0. In that environment:

```bash
git clone https://github.com/kingfloydz/nonprehensile.git
cd nonprehensile
python3 -m pip install -e .
python3 src/nonprehensile/pipeline.py --gpus 0 --num-envs 4096 --root runs/main
```

Several GPUs run independent experiments concurrently (one process per GPU):

```bash
python3 src/nonprehensile/pipeline.py --gpus 0 1 2 3 --num-envs 8192 --root runs/h200 \
  --search-iterations 300 --final-iterations 2000 --bank-size 8192 --workers 16
```

An absolute script path also works from any working directory. Relative output
paths are resolved against the working directory. Training and test subprocesses
use the same Python interpreter as the pipeline (`sys.executable`). With the
dependencies already installed in system Python, editable installation is optional.

The full pipeline performs **145 search runs + 23 final runs**. Sticking E is omitted
because it has no startup transition and would duplicate D. Each PPO iteration
contains `32 * num_envs` transitions. Weight selection uses highest validation
success rate, then lowest failure-penalized time; never shaped training return.
F has no physics-reward weights to search: each of its four final runs trains a
teacher for the final PPO budget, then distills a student. To run F only:

```bash
python3 src/nonprehensile/pipeline.py --groups F --gpus 0 --num-envs 4096 --root runs/f
```

Use `--groups A F` for the task-only baseline versus F, or `--groups A B C D E`
to retain only the original ablations.
The default search/final budgets are intentionally configurable. "Best" means
best on validation among the searched candidates, not globally optimal weights.

Outputs:

```
runs/main/
  banks/{train,validation,test}.npz
  search/<task>/<group>/<trial>/{config.json,policy.pt,validation.json,learning.jsonl}
  best/<task>/<group>/{config.json,policy.pt,validation.json,test.json,selected_from.txt}
  best/<task>/F/{teacher.pt,teacher_validation.json,teacher_test.json}
  best/<task>/F/{policy.pt,student_actor.pt,distillation.jsonl,distill_config.json}
  results.csv
```

`best` contains each group's final model. F's `policy.pt` is the student;
`teacher.pt` retains the constrained policy. F does not have `selected_from.txt`.
`results.csv` distinguishes `direct`, `teacher`, and `student` stages. Search
logs/checkpoints remain available for auditing. Validation and test use fixed
first-episode batches, so faster failures do not get overrepresented. Test results
include normalized acceleration noise standard deviations 0, .01, .03, .05.
`constraint_violation_rate` measures nominal commanded actions before disturbance
and before terminal/reset transitions, using normalized halfspaces/equalities
with tolerance 1e-4. `mapping_failure_rate` is the fraction of episodes terminated
by an unverified teacher mapping; it is separate from physical failures.

## Experiment

| Setting | Value |
|---|---|
| Tray | 0.80 x 0.80 m |
| Box length / width | independent U(0.06, 0.30) m |
| Box height | U(0.05, 0.40) m |
| Static = kinetic friction | U(0.10, 0.80) |
| Density | 500 kg/m^3; mass and inertia change with dimensions |
| Action | six tray accelerations, each angular component +/-30 rad/s^2, linear +/-15 m/s^2 |
| Tray speed | <=2 m/s, angular speed <=3 rad/s; exceeding a limit terminates the episode |
| Control / physics | 20 ms / 1 ms |
| Timeout | 5 s |
| Relative acceleration | beta >= 0 for all moving-mode analytic sets |

Each split contains independent uniform samples of dimensions and friction.
An episode samples a row from its split's **finite randomized bank**, rather than
drawing a fresh continuous parameter and running CPU double description on reset.
Increase `--bank-size` for wider empirical coverage. Banks are disjoint by RNG seed
(101/202/303); training policy seed is always 1. No unseen test data select weights.

| Task | Success | Forbidden motion |
|---|---|---|
| sticking | tray moves +0.5 m along world x and stops | box translation/rotation relative to tray |
| sliding | box first reaches +0.2 m along tray x | transverse sliding, relative rotation |
| tipping | box rotates about its +x bottom edge until the next face first touches | edge translation, rotation away from its y-axis |
| twisting | box first reaches +90 degrees about its normal COM axis | COM translation, tilt |

All start at relative rest. Sliding starts at x=-0.1 m and ends at x=+0.1 m;
tipping also starts at x=-0.1 m to leave room in its falling direction.
The selected task fixes direction; policies are trained independently per task.
Sliding/twisting end at first passage, **not at rest**, because beta >= 0 excludes
braking. Tipping ends at first new-face contact; post-impact settling is not scored.
All tasks reject separation, tray-speed violations, and leaving the tray.
Motion tolerances are 3 mm and 3 degrees; the sticking endpoint tolerance is 5 mm
with speeds below .03 m/s and .10 rad/s.

## Rewards

All groups share `-dt + 10*success - 10*failure`; timeout is failure. A common
potential term `gamma * Phi(next) - Phi(now)` uses negative normalized goal error
(plus a small speed term for transport), with zero terminal potential. This gives
all baselines goal-progress feedback without changing the discounted task optimum;
it is fixed across groups and is not part of the reward-weight search.
Physics costs are multiplied by dt, preventing dependence on reward sampling rate.

| Group | Additional penalty | Search weights |
|---|---|---|
| A | none | one setting |
| B | normalized unwanted translation/rotation | conventional: .1, 1, 10 |
| C | distance^2, after entering the moving mode | distance: .1, 1, 10 |
| D | C + softplus((.05 - relative_margin)/.05) | distance: .1, 1, 10; margin: .01, .1, 1 |
| E | D + startup-branch cost continuously until physical mode entry | D grid x startup multiplier: .3, 1, 3 |
| F | none; constrain teacher actions, then distill | no reward-weight search |

For sticking, C/D apply immediately. E uses the zero-relative-velocity startup
branch before entry, then the mode-interior branch. Entry is latched from measured
positive relative speed (> .02 m/s sliding, > .10 rad/s tipping/twisting), not from
analytic feasibility. These thresholds operationalize the near-rest startup state.
Beta=0 is deliberately included; startup feasibility alone does not force entry.
The task's time/success reward must still encourage starting and finishing.

## Shared observation (36 dimensions)

All groups, teacher and student receive the same state information and fixed
normalization. There are no method-specific inputs or running observation statistics.
Lengths are in meters and all relative quantities are expressed in the tray frame.

| Indices (zero-based) | Observation | Scaling | Dimensions |
|---|---|---|---|
| 0:3 | current object COM relative to tray origin | / 0.4 | 3 |
| 3:12 | object-to-tray rotation matrix, row-major | none | 9 |
| 12:15 | relative COM velocity | / 2 | 3 |
| 15:18 | relative angular velocity | / 3 | 3 |
| 18:21 | tray world linear velocity expressed in tray frame | / 2 | 3 |
| 21:24 | tray angular velocity | / 3 | 3 |
| 24:27 | gravity expressed in tray frame | / 9.81 | 3 |
| 27:30 | box dimensions | parameter ranges to [-1,1] | 3 |
| 30:31 | friction coefficient | [0.1,0.8] to [-1,1] | 1 |
| 31:34 | goal error | task-specific, below | 3 |
| 34:35 | latched physical mode-entry flag | 0 or 1 | 1 |
| 35:36 | elapsed episode steps | / 250 | 1 |

Transport uses the world target displacement rotated into the tray frame / 0.5.
Sliding uses `[remaining x displacement / 0.2, 0, 0]`; tipping/twisting use
`[remaining pitch/yaw / (pi/2), 0, 0]`. Velocity inputs are not clipped. Absolute
world pose and previous action are omitted; current COM position is retained so
contact lever arms and tray boundaries remain observable. Geometry, mass (fixed
density), friction and task goals are equally available to every method. This is
a full-state method comparison, not a perception or parameter-estimation test.
Previous **46-input checkpoints require retraining**.

## F: constrained PPO teacher → unconstrained student

For each state, include the action bounds in the normalized acceleration set:

$$\mathcal P=\{z:Az\le b,\ Ez=e,\ -1\le z\le1\},\qquad
\chi=\operatorname{diag}(30,30,30,15,15,15)z.$$

Compute an orthonormal nullspace basis $Q$ of $E$ and a verified relative-interior
point $z_0$. With $G=AQ$ and $h=b-Az_0>0$ on nonconstant rows, use

$$p(u)=\max_i\frac{G_i u}{h_i},\qquad
z_A=z_0+Q\frac{u}{1+p(u)},\qquad u\sim\mathcal N(\mu_\theta(o),\Sigma_\theta).$$

The bounded polytope makes this a bijection from the latent space to its relative
interior, provided the declared equalities span its affine hull. The latent
dimension is **6/4/6/4** for sticking/sliding/tipping/twisting in this contact model.
PPO stores and scores the **raw Gaussian sample** $u$, not the mapped acceleration.
The state-only map cancels from the likelihood ratio; no differentiation through
geometry or change-of-variables term is needed. The entropy bonus is latent
Gaussian entropy, not entropy in physical acceleration coordinates. Evaluation
and distillation use the **mapped latent mean**, not a noisy rollout sample.

The interior point uses a fixed-budget batched phase-I log barrier (default 32
Newton iterations, at most 7-by-7 systems). It starts deterministically from the
current state, without a history-dependent warm start. Padded/constant facets are
masked; every center and action is checked against the inequalities and equalities.
Rank change, collapsed interior, numerical failure or infeasibility terminates
the affected episode and increments the mapping-failure metric. Failure to find a
certificate is **not proof that the polytope is empty**. No action is presented as
feasible after a failed certificate; the batched simulator uses a zero placeholder
for that discarded terminal transition. `env.mapping_iterations` controls the
solver budget. Full float32 matmul precision is used for the equality checks.

| Stage | Default | Output |
|---|---|---|
| Teacher PPO | same task reward and final PPO iterations as A | `teacher.pt` |
| Behavior cloning | 1,000,000 teacher-rollout transitions, 10 epochs | student initialization |
| DAgger | 10 rounds × 100,000 student-rollout transitions, 3 epochs on accumulated data | `policy.pt` |
| Deployment | 36 → 256 → 256 → 6, tanh activations and tanh output | `student_actor.pt` |

Distillation minimizes normalized-action MSE, batch size 4096, Adam learning rate
3e-4. All datasets stay on the selected device. Teacher labels are queried at the
student's actual visited states; invalid labels are excluded and those rollouts
terminated. Rejected-label fractions are logged. Config keys in `distill` are
`samples`, `epochs`, `batch_size`, `learning_rate`, `dagger_rounds`, `dagger_samples`,
and `dagger_epochs`. A direct `train --config ...` run can override these budgets.
F's additional simulation and distillation cost must be included in comparisons;
equal teacher PPO iterations do not imply equal total compute or transitions.

Deployment requires only the exported actor, with **no feasible-set mapping**:

```python
actor = torch.jit.load("student_actor.pt", map_location="cuda:0")
with torch.inference_mode():
    normalized_action = actor(observation)  # [batch, 36] -> [batch, 6]
    acceleration = normalized_action * observation.new_tensor([30, 30, 30, 15, 15, 15])
```

The student has no hard feasibility guarantee. The teacher's certificate is an
instantaneous analytic-model property, not a guarantee for the whole 20 ms physics
interval or the entire episode. Physical contact/mode deviations remain measured
by the unchanged independent simulator. Report both teacher and student success,
physical failure, nominal constraint violations, and mapping failures.

## Physics and geometry

The payload is simulated by MuJoCo-Warp in the accelerating tray coordinate frame.
The tray geometry is static in that frame. Per-substep fictitious forces and
torques account for translation, angular acceleration, centrifugal, Coriolis and
gyroscopic terms. Tray world pose/velocity are integrated separately. This avoids
servo tracking error or the assumption of a heavy tray, while contact impulses
and object motion remain independently solved by MuJoCo.

The CPU bank builder performs **V-to-H double description** by intersecting each
cone with f_z=1, converting vertices to facets with Qhull in its affine hull, and
homogenizing. Equality constraints are retained. Static contact uses an inscribed
8-direction friction polygon; sliding/twisting use fixed Coulomb rays. Twisting
uses the four-vertex contact approximation, not an exact continuous pressure model.

At runtime the wrench cone is shifted/rotated and pulled back through Newton-Euler
dynamics. For these models a contact-cone equality uniquely determines the scalar
beta. Eliminating it algebraically and adding beta>=0 is equivalent to projecting
`cone[V,-D]`, without rerunning double description for every GPU world/time step.
Action bounds are included in the final acceleration polyhedron.

Relative margin uses the normalized affine-hull projector and facet slacks.
Distance uses **48 fixed FISTA iterations** for the convex projection dual: it is
a numerical approximation, not an exact signed-distance formula. Iterations are
configurable in `EnvConfig`. Equality-normal distance is handled separately.
Analytic geometry is used by C/D/E rewards and F's teacher mapping, never as an
extra policy input. F uses actual projected mode velocities; unlike E it does not
zero them until the measured entry threshold. At exact rest these velocities
already vanish, giving the same startup dynamics.

All batches and physics operate on the selected GPU. Python loops are over PPO
steps, physics substeps and solver iterations, never over simulation worlds.
CPU double description runs only when generating parameter banks. Geometry and
reward costs must be included when reporting training throughput.
The official mjlab `Simulation`, configuration, model randomization and reset
interfaces are reused. The existing compact PPO loop is retained to score latent
teacher actions explicitly and keep A-E comparable. Teacher/baseline actors,
critics and the deployment student all use the official `rsl_rl.modules.MLP`
from mjlab's RSL-RL dependency; no local MLP implementation is maintained.

## Source map

| File | Responsibility |
|---|---|
| `bank.py` | randomized parameter banks and double description |
| `geometry.py` | beta elimination, acceleration constraints, margin/distance |
| `mapping.py` | normalized constraints, affine-hull gauge map, batched interior solver |
| `env.py` | mjlab simulation, shared 36D observation, four tasks, A-F rewards |
| `ppo.py` | PPO training, fixed-sample validation/test |
| `distill.py` | teacher action labels, BC, DAgger, student export |
| `pipeline.py` | weight grid, GPU scheduling, best-setting selection |

Development tests (not part of the training pipeline):

```bash
pip install pytest
pytest tests -q
```

The mathematical tests compare DD membership, scalar elimination and gauge-map
relative feasibility against independent linear programs; they also check inverse
mapping, invalid-set rejection and batch independence. Physics tests check
world-frame free fall, observation frames, failure resets, and a small real-physics
PPO → BC → DAgger → exported-student run. H200 throughput and
policy learning results must be measured on the training server; no trained
success rates are claimed by this repository.
