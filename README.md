# Nonprehensile tray RL

Four independent tasks, five reward ablations, one training seed (`1`). Built on
the installed **mjlab 1.6.0 / MuJoCo-Warp** simulation API with batched PyTorch PPO.
The server pipeline prepares geometry, searches weights, selects on held-out
validation, retrains each selected setting, and exports held-out test results.
There are no automatic retries, CPU fallbacks, or environment-detection scripts.

## H200

Use Linux, Python 3.12, an NVIDIA driver compatible with the CUDA build of PyTorch,
and GPU-enabled mjlab 1.6.0. In that environment:

```bash
git clone https://github.com/kingfloydz/nonprehensile.git
cd nonprehensile
pip install -e .
npt pipeline --gpus 0 --num-envs 4096 --root runs/main
```

Several GPUs run independent experiments concurrently (one process per GPU):

```bash
npt pipeline --gpus 0 1 2 3 --num-envs 8192 --root runs/h200 \
  --search-iterations 300 --final-iterations 2000 --bank-size 8192 --workers 16
```

The pipeline performs **145 search runs + 19 final runs**. Sticking E is omitted
because it has no startup transition and would duplicate D. Each PPO iteration
contains `32 * num_envs` transitions. Weight selection uses highest validation
success rate, then lowest failure-penalized time; never shaped training return.
The default search/final budgets are intentionally configurable. "Best" means
best on validation among the searched candidates, not globally optimal weights.

Outputs:

```
runs/main/
  banks/{train,validation,test}.npz
  search/<task>/<group>/<trial>/{config.json,policy.pt,validation.json,learning.jsonl}
  best/<task>/<group>/{config.json,policy.pt,validation.json,test.json,selected_from.txt}
  results.csv
```

`best` contains each group's selected weight settings and final model. Search
logs/checkpoints remain available for auditing. Validation and test use fixed
first-episode batches, so faster failures do not get overrepresented. Test results
include normalized acceleration noise standard deviations 0, .01, .03, .05.

## Experiment

| Setting | Value |
|---|---|
| Tray | 0.60 x 0.60 m |
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

For sticking, C/D apply immediately. E uses the zero-relative-velocity startup
branch before entry, then the mode-interior branch. Entry is latched from measured
positive relative speed (> .02 m/s sliding, > .10 rad/s tipping/twisting), not from
analytic feasibility. These thresholds operationalize the near-rest startup state.
Beta=0 is deliberately included; startup feasibility alone does not force entry.
The task's time/success reward must still encourage starting and finishing.

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
The policy receives the same 46 observations in all groups; analytic geometry is
used only by the reward, not as privileged policy input for C/D/E.

All batches and physics operate on the selected GPU. Python loops are over PPO
steps, physics substeps and solver iterations, never over simulation worlds.
CPU double description runs only when generating parameter banks. Geometry and
reward costs must be included when reporting training throughput.

## Source map

| File | Responsibility |
|---|---|
| `bank.py` | randomized parameter banks and double description |
| `geometry.py` | beta elimination, acceleration constraints, margin/distance |
| `env.py` | mjlab simulation, randomization, four tasks, A-E rewards |
| `ppo.py` | PPO training, fixed-sample validation/test |
| `pipeline.py` | weight grid, GPU scheduling, best-setting selection |

Development tests (not part of the training pipeline):

```bash
pip install pytest
pytest tests -q
```

The mathematical tests compare DD membership and scalar elimination against
independent linear programs. Physics tests check world-frame free fall under an
accelerating/rotating tray frame and all four reward modes. H200 throughput and
policy learning results must be measured on the training server; no trained
success rates are claimed by this repository.
