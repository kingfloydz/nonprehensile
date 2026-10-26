import numpy as np
import pytest
import torch
from scipy.optimize import linprog

from nonprehensile.bank import TASKS, double_description, generators
from nonprehensile.geometry import acceleration_constraints, measures


def test_double_description_matches_generator_membership():
    rng = np.random.default_rng(7)
    for task in TASKS:
        for _ in range(3):
            v = generators(rng.uniform(0.06, 0.3, 3), rng.uniform(0.1, 0.8), task)
            h, e = double_description(v)
            assert (h @ v).max() < 1e-7
            assert np.max(np.abs(e @ v), initial=0) < 1e-7
            for _ in range(12):
                w = v @ rng.uniform(0, 1, v.shape[1])
                w += rng.normal(0, 0.2, 6) * rng.integers(0, 2)
                primal = linprog(
                    np.zeros(v.shape[1]), A_eq=v, b_eq=w, bounds=(0, None), method="highs"
                ).success
                dual = np.all(h @ w <= 1e-7) and np.all(np.abs(e @ w) <= 1e-7)
                assert primal == dual


@pytest.mark.parametrize("speed", [0.0, 0.4, -0.04])
@pytest.mark.parametrize("dt", [0.02, 0.05])
@pytest.mark.parametrize("startup", [False, True])
def test_scalar_elimination_matches_lifted_dynamics(speed, dt, startup):
    size = torch.tensor([[0.16, 0.12, 0.20]], dtype=torch.float64)
    mass = torch.tensor([1.92], dtype=torch.float64)
    inertia = torch.diag_embed(
        mass[:, None] / 12 * (size.square().sum(1, keepdim=True) - size.square())
    )
    c = torch.tensor([[0.0, 0.0, 0.1]], dtype=torch.float64)
    zero = torch.zeros((1, 3), dtype=torch.float64)
    gravity = torch.tensor([[0.0, 0.0, -9.81]], dtype=torch.float64)
    eye = torch.eye(3, dtype=torch.float64)[None]
    # Constants at the common zero-velocity state; compare original and eliminated systems.
    from nonprehensile.math import skew

    cross = skew(c)
    bmat = torch.cat(
        (
            torch.cat(
                (inertia - mass[:, None, None] * cross @ cross, mass[:, None, None] * cross), -1
            ),
            torch.cat((-mass[:, None, None] * cross, mass[:, None, None] * eye), -1),
        ),
        -2,
    )[0].numpy()
    rng = np.random.default_rng(3)
    for task in TASKS[1:]:
        v = generators(size[0].numpy(), 0.5, task)
        h, e = double_description(v)
        uv = {
            "sliding": np.array([1.0, 0, 0]),
            "tipping": np.array([0.1, 0, 0.08]),
            "twisting": np.zeros(3),
        }[task]
        uw = {
            "sliding": np.zeros(3),
            "tipping": np.array([0.0, 1, 0]),
            "twisting": np.array([0.0, 0, 1]),
        }[task]
        f, eq = acceleration_constraints(
            task,
            torch.tensor(h)[None],
            torch.tensor(e)[None],
            c,
            inertia,
            mass,
            zero,
            gravity,
            torch.tensor(speed * uv)[None],
            torch.tensor(speed * uw)[None],
            size,
            eye,
            zero,
            torch.tensor([startup]),
            dt=dt,
        )
        assert eq.shape[1] == (0 if task == "tipping" else 2)
        # Independent lifted Newton-Euler model, including the tipping centripetal term.
        ar = np.cross(uw, uv) * (0 if startup else speed**2)
        force = mass.item() * (ar - gravity[0].numpy())
        b = np.r_[np.cross(c[0].numpy(), force), force]
        lower = -speed / (2 * dt)
        d = np.r_[inertia[0].numpy() @ uw + mass.item() * np.cross(c[0], uv), mass.item() * uv]
        for _ in range(15):
            # Include feasible samples by constructing chi from the lifted system.
            chi = np.linalg.solve(
                bmat, v @ rng.uniform(0, 1, v.shape[1]) - d * (lower + rng.uniform(-1, 3)) - b
            )
            if rng.random() < 0.5:
                chi += rng.normal(0, 1, 6)
            original = linprog(
                np.zeros(v.shape[1] + 1),
                A_eq=np.column_stack((v, -d)),
                b_eq=bmat @ chi + b,
                bounds=[(0, None)] * v.shape[1] + [(lower, None)],
                method="highs",
            ).success
            x = np.r_[chi, 1]
            eliminated = np.all(f[0].numpy() @ x < 1e-6) and np.all(abs(eq[0].numpy() @ x) < 1e-6)
            assert original == eliminated


def test_margin_and_projection_on_affine_halfspace():
    # x0 <= 0, x1 = 0 in fixed metric coordinates; the enlarged box is [-2,2].
    inequalities = torch.tensor([[[1 / 30, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
    eq = torch.tensor([[[0.0, 1 / 30, 0.0, 0.0, 0.0, 0.0, 0.0]]])
    margin, distance = measures(inequalities, eq, torch.tensor([[0.15, 0.2, 0, 0, 0, 0.0]]), 256)
    torch.testing.assert_close(margin, torch.tensor([-0.3]), atol=1e-5, rtol=0)
    torch.testing.assert_close(distance, torch.tensor([0.5]), atol=1e-4, rtol=0)
    margin, distance = measures(inequalities, eq, torch.tensor([[-0.15, 0, 0, 0, 0, 0.0]]))
    torch.testing.assert_close(margin, torch.tensor([0.3]), atol=1e-5, rtol=0)
    torch.testing.assert_close(distance, torch.tensor([0.0]), atol=1e-5, rtol=0)


def test_expanded_action_box_uses_original_margin_units():
    empty = torch.zeros(1, 0, 7)
    margin, distance = measures(empty, empty, torch.tensor([[0.75, 0, 0, 0, 0, 0.0]]))
    torch.testing.assert_close(margin, torch.tensor([0.5]))
    torch.testing.assert_close(distance, torch.tensor([0.0]))


def test_physical_distance_does_not_change_with_policy_scaling():
    # Same physical point, same halfspace and fixed metric, different action boxes.
    h = torch.tensor([[[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]])
    e = torch.zeros(1, 0, 7)
    physical = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
    results = []
    for scale in ((2.0, 2.0, 2.0, 2.0, 1.0, 1.0), (2.0, 2.0, 30.0, 5.0, 2.0, 2.0)):
        results.append(
            measures(h, e, physical / torch.tensor(scale), iterations=256, action_scale=scale)
        )
    for margin, distance in results:
        torch.testing.assert_close(margin, torch.tensor([-1 / 15]), atol=1e-5, rtol=0)
        torch.testing.assert_close(distance, torch.tensor([1 / 15]), atol=1e-5, rtol=0)
