"""Pull back DD contact constraints with a one-step directional speed bound."""

import torch

from .math import mv, skew
from .settings import METRIC_SCALE

# Legacy standalone-call default. Training always passes its per-task limits;
# reward geometry keeps fixed physical perturbation units independently.
SCALE = (60.0, 60.0, 60.0, 30.0, 30.0, 30.0)


def acceleration_constraints(
    task,
    h,
    e,
    c,
    inertia,
    mass,
    omega,
    gravity,
    relative_velocity,
    relative_omega,
    size,
    object_rotation,
    origin,
    startup,
    dt=0.02,
):
    if dt <= 0:
        raise ValueError("The control period must be positive")
    n = c.shape[0]
    eye = torch.eye(3, device=c.device, dtype=c.dtype).expand(n, 3, 3)
    zero = torch.zeros_like(c)
    axis = zero.clone()
    axis[:, 1 if task == "tipping" else 2] = 1
    vr, wr = relative_velocity, relative_omega
    if task == "sticking":
        vr, wr, uv, uw, ar = zero, zero, zero, zero, zero
    elif task == "sliding":
        uv = zero.clone()
        uv[:, 0] = 1
        vr = uv * relative_velocity[:, :1]
        wr, uw, ar = zero, zero, zero
    elif task == "tipping":
        edge = origin.clone()
        edge[:, 0] += size[:, 0] / 2
        radius = c - edge
        uv, uw = torch.linalg.cross(axis, radius), axis
        speed = (relative_omega * axis).sum(-1, keepdim=True)
        vr, wr = speed * uv, speed * axis
        ar = speed.square() * torch.linalg.cross(axis, uv)
    else:
        uv, uw, ar = zero, axis, zero
        vr = zero
        wr = axis * relative_omega[:, 2:3]
    vr = torch.where(startup[:, None], zero, vr)
    wr = torch.where(startup[:, None], zero, wr)
    ar = torch.where(startup[:, None], zero, ar)
    cross = skew(c)
    m = mass[:, None, None]
    bmat = torch.cat(
        (
            torch.cat((inertia - m * cross @ cross, m * cross), -1),
            torch.cat((-m * cross, m * eye), -1),
        ),
        -2,
    )
    k = torch.linalg.cross(omega, torch.linalg.cross(omega, c))
    k = k + 2 * torch.linalg.cross(omega, vr) + ar - gravity
    total = omega + wr
    torque = mv(inertia, torch.linalg.cross(omega, wr))
    torque += torch.linalg.cross(total, mv(inertia, total)) + mass[:, None] * mv(cross, k)
    bias = torch.cat((torque, mass[:, None] * k), -1)
    direction = torch.cat((mv(inertia, uw) + mass[:, None] * mv(cross, uv), mass[:, None] * uv), -1)
    # Wrench reference is the translating bottom-center, except for the fixed tip edge.
    ref = origin if task == "tipping" else c.clone()
    if task != "tipping":
        ref[:, 2] = 0
    rot = object_rotation if task == "twisting" else eye
    rt = rot.transpose(-1, -2)
    transform = torch.zeros((n, 6, 6), device=c.device, dtype=c.dtype)
    transform[:, :3, :3] = rt
    transform[:, :3, 3:] = -rt @ skew(ref)
    transform[:, 3:, 3:] = rt
    wmap = transform @ torch.cat((bmat, bias[:, :, None]), -1)
    direction = mv(transform, direction)
    if task == "sticking":
        inequalities = h @ wmap
        equalities = wmap[:, :0]
    else:
        ed = mv(e, direction)
        ew = e @ wmap
        # One mode acceleration is determined by a contact-cone equality.
        beta = -(ed[:, :, None] * ew).sum(1) / ed.square().sum(1, keepdim=True)
        corrected = wmap + direction[:, :, None] * beta[:, None, :]
        # beta >= -v_parallel/(2*dt). Use measured signed speed even when E
        # approximates the startup dynamics with zero relative velocity.
        speed = (
            relative_velocity[:, 0]
            if task == "sliding"
            else relative_omega[:, 1 if task == "tipping" else 2]
        )
        direction_bound = -beta.clone()
        direction_bound[:, 6] -= speed / (2 * dt)
        inequalities = torch.cat((h @ corrected, direction_bound[:, None]), 1)
        # Remove the consumed equality explicitly, rather than retaining a row
        # of roundoff errors that pinv could mistake for a physical constraint.
        left, _, _ = torch.linalg.svd(ed[:, :, None], full_matrices=True)
        equalities = left[:, :, 1:].transpose(1, 2) @ ew
    return inequalities, equalities


def measures(inequalities, equalities, action, iterations=48, action_scale=SCALE):
    """Relative slack and fixed-iteration Euclidean projection distance.

    Coordinates use fixed METRIC_SCALE units, independently of action bounds.
    The input action is policy-normalized by action_scale. FISTA solves the
    halfspace projection dual on GPU; no per-world optimizer or host synchronization.
    """
    n = action.shape[0]
    scale = action.new_tensor(METRIC_SCALE)
    bounds = torch.as_tensor(action_scale, device=action.device, dtype=action.dtype) / scale
    action = action * bounds
    eye = torch.eye(6, device=action.device).expand(n, 6, 6)
    a = inequalities[:, :, :6] * scale
    b = -inequalities[:, :, 6]
    a = torch.cat((a, eye, -eye), 1)
    b = torch.cat((b, bounds.repeat(2).expand(n, 12)), 1)
    if equalities.shape[1]:
        eq = equalities[:, :, :6] * scale
        rhs = -equalities[:, :, 6]
        inverse = torch.linalg.pinv(eq, rtol=1e-5)
        projection = eye - inverse @ eq
        perpendicular = mv(inverse, mv(eq, action) - rhs)
    else:
        projection = eye
        perpendicular = torch.zeros_like(action)
    parallel = action - perpendicular
    tangent = a @ projection
    length = tangent.norm(dim=-1)
    active = length > 1e-6
    slack = (b - mv(a, parallel)) / length.clamp_min(1e-6)
    margin = torch.where(active, slack, torch.inf).amin(1)
    normals = tangent / length.clamp_min(1e-6)[:, :, None]
    normals = torch.where(active[:, :, None], normals, 0.0)
    slack = torch.where(active, slack, 0.0)
    lipschitz = torch.linalg.eigvalsh(normals.transpose(1, 2) @ normals)[:, -1]
    dual = torch.zeros_like(slack)
    accelerated = dual.clone()
    t = 1.0
    for _ in range(iterations):
        gradient = mv(normals, mv(normals.transpose(1, 2), accelerated)) + slack
        update = (accelerated - gradient / lipschitz[:, None]).clamp_min(0)
        t_new = (1 + (1 + 4 * t * t) ** 0.5) / 2
        accelerated = update + ((t - 1) / t_new) * (update - dual)
        dual, t = update, t_new
    correction = mv(normals.transpose(1, 2), dual)
    distance = (perpendicular.square().sum(1) + correction.square().sum(1)).sqrt()
    return margin, distance
