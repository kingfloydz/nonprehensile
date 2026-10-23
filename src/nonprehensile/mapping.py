"""State-only affine-hull gauge map. All online solves are small GPU batches."""

import torch

from .geometry import SCALE
from .math import mv


def normalized_constraints(inequalities, equalities):
    """Az <= b, Ez = e in normalized acceleration coordinates, including bounds."""
    scale = inequalities.new_tensor(SCALE)
    n = len(inequalities)
    eye = torch.eye(6, device=scale.device, dtype=scale.dtype).expand(n, 6, 6)
    a = torch.cat((inequalities[..., :6] * scale, eye, -eye), 1)
    b = torch.cat((-inequalities[..., 6], scale.new_ones(n, 12)), 1)
    e, rhs = equalities[..., :6] * scale, -equalities[..., 6]
    length = a.norm(dim=-1).clamp_min(1e-8)
    eq_length = e.norm(dim=-1).clamp_min(1e-8)
    return a / length[..., None], b / length, e / eq_length[..., None], rhs / eq_length


def violation(a, b, e, rhs, action):
    inequality = (mv(a, action) - b).clamp_min(0).amax(-1)
    equality = (mv(e, action) - rhs).abs().sum(-1)
    return torch.maximum(inequality, equality)


def interior_point(g, h, active, iterations=32):
    """Fixed-budget phase-I log barrier; correctness comes from residual checks.

    Minimize t - mu*mean(log(h-Gy+t)) + 1e-4*||(y,t)||^2/2.
    The auxiliary t gives an initially strict feasible point even when y=0 is
    outside. No warm start: the mapping depends only on the current state.
    A failed certificate means 'no verified interior', not proof of emptiness.
    """
    n, _, r = g.shape
    j = torch.cat((-g, active[..., None].to(g.dtype)), -1)
    x = g.new_zeros(n, r + 1)
    x[:, -1] = torch.where(active, 1 - h, -torch.inf).amax(-1)
    identity = torch.eye(r + 1, device=g.device, dtype=g.dtype)
    count = active.sum(-1).clamp_min(1)
    linear = g.new_zeros(r + 1)
    linear[-1] = 1
    for k in range(iterations):
        mu = 0.1 ** min(3, (4 * k) // iterations)
        weight = active * (mu / count[:, None])
        slack = (h + mv(j, x)).clamp_min(1e-9)
        gradient = linear + 1e-4 * x - mv(j.transpose(1, 2), weight / slack)
        hessian = j.transpose(1, 2) @ ((weight / slack.square())[..., None] * j)
        hessian = hessian + 1e-4 * identity
        # solve_ex avoids the CUDA-to-host error check of torch.linalg.solve.
        direction, _ = torch.linalg.solve_ex(hessian, -gradient[..., None], check_errors=False)
        direction = direction.squeeze(-1)
        change = mv(j, direction)
        limit = torch.where(change < 0, -0.99 * slack / change, torch.inf).amin(-1)
        step = limit.clamp(max=1)
        objective = x[:, -1] + 5e-5 * x.square().sum(-1) - (weight * slack.log()).sum(-1)
        slope = (gradient * direction).sum(-1)
        # Batched Armijo backtracking; no per-world loop or CPU optimizer.
        for _ in range(8):
            trial = x + step[:, None] * direction
            trial_slack = (h + mv(j, trial)).clamp_min(1e-9)
            cost = trial[:, -1] + 5e-5 * trial.square().sum(-1)
            cost -= (weight * trial_slack.log()).sum(-1)
            step = torch.where(cost <= objective + 0.01 * step * slope, step, step * 0.5)
        x += step[:, None] * direction
    return x[:, :r]


class GaugeMap:
    """u in R^r -> z0 + Q u/(1 + max_i (G_i u / h_i))."""

    @torch.no_grad()
    def __init__(self, inequalities, equalities, iterations=32):
        self.a, self.b, self.e, self.rhs = normalized_constraints(inequalities, equalities)
        a, b, e, rhs = self.a, self.b, self.e, self.rhs
        n, k = e.shape[:2]
        if k:
            left, singular, right = torch.linalg.svd(e, full_matrices=True)
            self.q = right[:, k:].transpose(1, 2)
            point = mv(
                right[:, :k].transpose(1, 2),
                mv(left.transpose(1, 2), rhs) / singular.clamp_min(1e-8),
            )
            rank_ok = singular[:, -1] > 1e-5 * singular[:, 0].clamp_min(1)
            # Fix column signs to avoid arbitrary sign flips across SVD calls.
            pivot = self.q.abs().argmax(1, keepdim=True)
            self.q = self.q * self.q.gather(1, pivot).sign()
        else:
            self.q = torch.eye(6, device=a.device, dtype=a.dtype).expand(n, 6, 6)
            point = a.new_zeros(n, 6)
            rank_ok = torch.ones(n, device=a.device, dtype=torch.bool)
        g, h = a @ self.q, b - mv(a, point)
        length = g.norm(dim=-1)
        active = length > 1e-6
        constant_ok = (active | (h >= -1e-5)).all(-1)
        self.g = torch.where(active[..., None], g / length.clamp_min(1e-6)[..., None], 0)
        h = torch.where(active, h / length.clamp_min(1e-6), 1)
        y = interior_point(self.g, h, active, iterations)
        self.center = point + mv(self.q, y)
        self.slack = h - mv(self.g, y)
        self.valid = rank_ok & constant_ok & torch.isfinite(self.center).all(-1)
        self.valid &= torch.where(active, self.slack, torch.inf).amin(-1) > 1e-5
        self.valid &= violation(a, b, e, rhs, self.center) < 5e-5

    @torch.no_grad()
    def __call__(self, raw):
        gauge = (mv(self.g, raw) / self.slack.clamp_min(1e-8)).amax(-1).clamp_min(0)
        action = (self.center + mv(self.q, raw / (1 + gauge[:, None]))).clamp(-1, 1)
        valid = self.valid & torch.isfinite(action).all(-1)
        valid &= violation(self.a, self.b, self.e, self.rhs, action) < 5e-5
        return torch.where(valid[:, None], action, 0.0), ~valid
