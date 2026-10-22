"""Offline V-to-H double description; no hull work in the simulation loop.

Every force generator has f_z=1. Intersect its cone with that plane, use
Qhull for the vertex-to-facet conversion in its affine hull, then homogenize.
"""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

TASKS = ("sticking", "sliding", "tipping", "twisting")


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


def _one(args):
    size, mu = args
    return [double_description(generators(size, mu, task)) for task in TASKS]


def build(path, count=8192, seed=101, workers=8):
    rng = np.random.default_rng(seed)
    sizes = rng.uniform([0.06, 0.06, 0.05], [0.30, 0.30, 0.40], size=(count, 3))
    friction = rng.uniform(0.1, 0.8, count)
    with ProcessPoolExecutor(workers) as pool:
        results = list(pool.map(_one, zip(sizes, friction), chunksize=16))
    arrays = dict(size=sizes.astype("float32"), mu=friction.astype("float32"))
    for k, task in enumerate(TASKS):
        nh = max(len(r[k][0]) for r in results)
        ne = len(results[0][k][1])
        h = np.zeros((count, nh, 6), dtype="float32")
        e = np.zeros((count, ne, 6), dtype="float32")
        for i, result in enumerate(results):
            hi, ei = result[k]
            h[i, : len(hi)] = hi
            e[i] = ei
        arrays[f"{task}_h"] = h
        arrays[f"{task}_e"] = e
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
