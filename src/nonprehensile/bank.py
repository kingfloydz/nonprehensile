"""Offline V-to-H double description; no hull work in the simulation loop.

Every force generator has f_z=1. Intersect its cone with that plane, use
Qhull for the vertex-to-facet conversion in its affine hull, then homogenize.
"""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from .settings import BENCHMARK_VERSION, SETTINGS, TASKS


def generators(size, mu, task):
    a, b, _ = size / 2
    points = np.array([[-a, -b, 0], [-a, b, 0], [a, -b, 0], [a, b, 0]])
    if task == "tipping":
        points = points[2:]
    if task in ("sticking", "tipping"):
        angles = np.arange(8) * np.pi / 4
        directions = np.column_stack((mu * np.cos(angles), mu * np.sin(angles), np.ones(8)))
        r = np.repeat(points, 8, axis=0)
        d = np.tile(directions, (len(points), 1))
    elif task == "sliding":
        r = points
        d = np.tile([-mu, 0, 1], (4, 1))
    else:
        r = points
        tangent = np.column_stack((-r[:, 1], r[:, 0], np.zeros(4)))
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
        d = np.array([0, 0, 1]) - mu * tangent
    return np.concatenate((np.cross(r, d), d), axis=1).T


def double_description(v):
    center = v.mean(axis=1)
    u, s, _ = np.linalg.svd(v - center[:, None], full_matrices=True)
    rank = np.count_nonzero(s > 1e-10)
    basis = u[:, :rank]
    hull = ConvexHull((v.T - center) @ basis)
    normals = hull.equations[:, :-1] @ basis.T
    offsets = hull.equations[:, -1] - normals @ center
    normals[:, 5] += offsets
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.unique(np.round(normals, 10), axis=0)
    normals = np.concatenate((normals, [[0, 0, 0, 0, 0, -1]]))
    u, s, _ = np.linalg.svd(v, full_matrices=True)
    equality = u[:, np.count_nonzero(s > 1e-10) :].T
    return normals, equality


def settled_pose(size, mu, tilt, offset, task):
    """Static initialization only; no reference trajectory enters policy training.

    Use the same contact model as the batched environment, and set mass/inertia
    before settling. Settling offline avoids advancing unrelated live episodes.
    """
    import mujoco
    from .model import XML

    model = mujoco.MjModel.from_xml_string(XML)
    model.opt.timestep = 0.0005
    model.opt.iterations, model.opt.ls_iterations = 30, 10
    model.geom_size[1] = size / 2
    model.geom_friction[1, 0] = mu
    model.body_mass[1] = 1.0
    model.body_inertia[1] = (np.sum(size**2) - size**2) / 12
    data = mujoco.MjData(model)
    mujoco.mj_setConst(model, data)
    data.qpos[:3] = [offset[0], offset[1], size[2] / 2]
    if task in ("sliding", "tipping"):
        data.qpos[0] -= 0.1
    data.qpos[3:7] = [1, 0, 0, 0]
    tray_rotation = Rotation.from_euler("xy", tilt, degrees=True)
    gravity = tray_rotation.inv().apply([0, 0, -9.81])
    for _ in range(160):
        data.qfrc_applied[:3] = gravity
        mujoco.mj_step(model, data)
    xyzw = tray_rotation.as_quat()
    return data.qpos.copy(), np.r_[xyzw[3], xyzw[:3]]


def _one(args):
    size, mu, tilt, offset, task = args
    h, e = double_description(generators(size, mu, task))
    qpos, quaternion = settled_pose(size, mu, tilt, offset, task)
    return h, e, qpos, quaternion


def build(path, count=8192, seed=101, workers=8):
    rng = np.random.default_rng(seed)
    arrays = dict(benchmark_version=np.array(BENCHMARK_VERSION))
    with ProcessPoolExecutor(workers) as pool:
        for task in TASKS:
            spec = SETTINGS[task]
            sizes = rng.uniform(spec.size_low, spec.size_high, size=(count, 3))
            friction = rng.uniform(*spec.friction, count)
            tilt = rng.uniform(-spec.initial_tilt_deg, spec.initial_tilt_deg, (count, 2))
            offset = rng.uniform(-spec.initial_xy_offset, spec.initial_xy_offset, (count, 2))
            results = list(
                pool.map(_one, zip(sizes, friction, tilt, offset, [task] * count), chunksize=16)
            )
            nh = max(len(r[0]) for r in results)
            ne = len(results[0][1])
            h = np.zeros((count, nh, 6), dtype="float32")
            e = np.zeros((count, ne, 6), dtype="float32")
            for i, (hi, ei, _, _) in enumerate(results):
                h[i, : len(hi)] = hi
                e[i] = ei
            for name, value in dict(
                size=sizes,
                mu=friction,
                h=h,
                e=e,
                qpos=[r[2] for r in results],
                quaternion=[r[3] for r in results],
            ).items():
                arrays[f"{task}_{name}"] = np.asarray(value, dtype="float32")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
