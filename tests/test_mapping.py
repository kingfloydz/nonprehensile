import numpy as np
import pytest
import torch
from scipy.optimize import linprog

from nonprehensile.bank import TASKS, double_description, generators
from nonprehensile.geometry import SCALE, acceleration_constraints
from nonprehensile.mapping import GaugeMap, violation
from nonprehensile.math import mv


@pytest.mark.parametrize("rank", [0, 2])
def test_gauge_bijection_and_batch_independence(rank):
    torch.manual_seed(4)
    torch.set_num_threads(2)
    n = 32
    center = torch.rand(n, 6, dtype=torch.float64) * 0.6 - 0.3
    a = torch.randn(n, 12, 6, dtype=torch.float64)
    b = mv(a, center) + 0.2 + torch.rand(n, 12, dtype=torch.float64)
    e = torch.randn(n, rank, 6, dtype=torch.float64)
    rhs = mv(e, center)
    scale = center.new_tensor(SCALE)
    f = torch.cat((a / scale, -b[..., None]), -1)
    # Banks pad facet rows with zero; these must not obstruct a strict interior.
    f = torch.cat((f, f.new_zeros(n, 3, 7)), 1)
    eq = torch.cat((e / scale, -rhs[..., None]), -1)
    mapping = GaugeMap(f, eq)
    assert mapping.valid.all()
    for magnitude in (0.0, 0.1, 1.0, 100.0):
        raw = torch.randn(n, 6 - rank, dtype=torch.float64) * magnitude
        action, invalid = mapping(raw)
        assert not invalid.any()
        assert (action.abs() <= 1 + 1e-7).all()
        assert (mv(a, action) <= b + 1e-7).all()
        torch.testing.assert_close(mv(e, action), rhs, atol=1e-7, rtol=0)
        y = mv(mapping.q.transpose(1, 2), action - mapping.center)
        gauge = (mv(mapping.g, y) / mapping.slack).amax(-1)
        torch.testing.assert_close(y / (1 - gauge[:, None]), raw, atol=1e-6, rtol=1e-6)
    single = GaugeMap(f[:1], eq[:1])
    torch.testing.assert_close(single.center, mapping.center[:1])
    torch.testing.assert_close(single.q, mapping.q[:1])


def test_mapping_rejects_empty_rank_change_and_collapsed_sets():
    # x0 <= -2 contradicts the action box; x0=0 encoded as two inequalities
    # has no interior in the declared affine hull. Neither may silently pass.
    f = torch.zeros(3, 2, 7)
    f[0, 0, 0], f[0, 0, 6] = 1 / SCALE[0], 2
    f[1, :, 0] = torch.tensor([1 / SCALE[0], -1 / SCALE[0]])
    f[2, 0, 6] = 1  # impossible constant row 0 <= -1
    mapping = GaugeMap(f, f[:, :0])
    _, invalid = mapping(torch.zeros(3, 6))
    assert invalid.all()
    e = torch.zeros(1, 2, 7)
    e[:, :, 0] = 1
    mapping = GaugeMap(f[:1] * 0, e)
    assert not mapping.valid.any()


def test_teacher_respects_task_acceleration_scale():
    # ax <= -1 m/s², with a narrow +/-2 m/s² action range.
    scale = (2.0, 2.0, 30.0, 2.0, 2.0, 2.0)
    h = torch.tensor([[[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]]])
    e = torch.zeros(1, 0, 7)
    mapping = GaugeMap(h, e, action_scale=scale)
    action, invalid = mapping(torch.zeros(1, 6))
    assert not invalid.any()
    physical = action * torch.tensor(scale)
    assert -2 < physical[0, 3] < -1


@pytest.mark.parametrize("task", TASKS)
def test_mode_mapping_matches_independent_phase_one(task):
    rng = np.random.default_rng(19)
    for _ in range(8):
        size = torch.tensor(
            rng.uniform([0.06, 0.06, 0.05], [0.30, 0.30, 0.40]), dtype=torch.float32
        )[None]
        mass = 500 * size.prod(1)
        inertia = torch.diag_embed(
            mass[:, None] / 12 * (size.square().sum(1, keepdim=True) - size.square())
        )
        h, e = double_description(generators(size[0].numpy(), rng.uniform(0.1, 0.8), task))
        zero = torch.zeros(1, 3)
        c = zero.clone()
        c[:, 2] = size[:, 2] / 2
        f, eq = acceleration_constraints(
            task,
            torch.tensor(h, dtype=torch.float32)[None],
            torch.tensor(e, dtype=torch.float32)[None],
            c,
            inertia,
            mass,
            zero,
            torch.tensor([[0.0, 0.0, -9.81]]),
            zero,
            zero,
            size,
            torch.eye(3)[None],
            zero,
            torch.tensor([False]),
        )
        mapping = GaugeMap(f, eq)
        # An independent CPU LP verifies strict relative feasibility; production
        # uses only the batched Torch solver, never this optimizer.
        a, b = mapping.a[0].numpy(), mapping.b[0].numpy()
        e, rhs = mapping.e[0].numpy(), mapping.rhs[0].numpy()
        tangent = (mapping.a @ mapping.q).norm(dim=-1)[0].numpy()
        phase = linprog(
            np.r_[np.zeros(6), -1.0],
            A_ub=np.c_[a, tangent],
            b_ub=b,
            A_eq=np.c_[e, np.zeros(len(e))] if len(e) else None,
            b_eq=rhs if len(e) else None,
            bounds=[(None, None)] * 6 + [(0, None)],
            method="highs",
        )
        assert phase.success and phase.x[-1] > 1e-4
        assert mapping.valid.all()
        action, invalid = mapping(torch.randn(1, 6 - eq.shape[1]))
        assert not invalid.any()
        assert violation(mapping.a, mapping.b, mapping.e, mapping.rhs, action).max() < 5e-5
