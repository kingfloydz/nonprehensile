import torch


def skew(v):
    x, y, z = v.unbind(-1)
    o = torch.zeros_like(x)
    return torch.stack((o, -z, y, z, o, -x, -y, x, o), -1).reshape(*v.shape[:-1], 3, 3)


def quat_mul(a, b):
    aw, av = a[..., :1], a[..., 1:]
    bw, bv = b[..., :1], b[..., 1:]
    return torch.cat(
        (aw * bw - (av * bv).sum(-1, keepdim=True), aw * bv + bw * av + torch.linalg.cross(av, bv)),
        -1,
    )


def quat_step(q, body_omega, dt):
    v = body_omega * dt
    angle = v.norm(dim=-1, keepdim=True)
    dq = torch.cat((torch.cos(angle / 2), 0.5 * torch.sinc(angle / (2 * torch.pi)) * v), -1)
    out = quat_mul(q, dq)
    return out / out.norm(dim=-1, keepdim=True)


def rotation(q):
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        -1,
    ).reshape(-1, 3, 3)


def mv(a, v):
    return (a @ v.unsqueeze(-1)).squeeze(-1)
