import numpy as np
from numba import njit, prange  # type: ignore


def init_boids(boids: np.ndarray, asp: float, v_range: tuple = (0., 1.)) -> np.ndarray:
    n = boids.shape[0]
    rng = np.random.default_rng()
    low, high = v_range
    boids[:, 0] = rng.uniform(0., asp, size=n)
    boids[:, 1] = rng.uniform(0., 1., size=n)
    alpha = rng.uniform(0, 2*np.pi, size=n)
    v = rng.uniform(low=low, high=high, size=n)
    c, s = np.cos(alpha), np.sin(alpha)
    boids[:, 2] = v * c
    boids[:, 3] = v * s
    return boids


@njit()
def directions(boids: np.ndarray,
               dt: float) -> np.ndarray:
    return np.hstack((boids[:, :2] - dt * boids[:, 2:4], boids[:, :2]))


@njit()
def norm(arr: np.ndarray):
    """
    Calculates norm via first axis
    param: a - (N, D)-shaped array of points, where D is the dimensions
           and N is points quantity
    returns: (N,)-shaped array of norms
    """
    return np.sqrt(np.sum(arr**2, axis=1))


@njit()
def mean_axis(arr, axis):
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.empty(arr.shape[1], dtype=arr.dtype)
        for i in range(len(result)):
            result[i] = np.mean(arr[:, i])
    else:
        result = np.empty(arr.shape[0], dtype=arr.dtype)
        for i in range(len(result)):
            result[i] = np.mean(arr[i, :])
    return result


@njit()
def median_axis(arr, axis):
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.empty(arr.shape[1], dtype=arr.dtype)
        for i in range(len(result)):
            result[i] = np.median(arr[:, i])
    else:
        result = np.empty(arr.shape[0], dtype=arr.dtype)
        for i in range(len(result)):
            result[i] = np.median(arr[i, :])
    return result


@njit()
def clip_mag(arr: np.ndarray,
             lims: tuple[float, float] = (0., 1.)):
    v = norm(arr)
    mask = v > 0
    v_clip = np.clip(v, *lims)
    arr[mask] *= (v_clip[mask] / v[mask]).reshape(-1, 1)


@njit()
def propagate(boids: np.ndarray, dt: float, v_range: tuple):
    boids[:, 2:4] += dt * boids[:, 4:6]
    clip_mag(boids[:, 2:4], lims=v_range)
    boids[:, 0:2] += dt * boids[:, 2:4]


@njit()
def periodic_walls(boids: np.ndarray, asp: float):
    boids[:, 0:2] %= np.array([asp, 1.])


@njit()
def wall_avoidance(boids: np.ndarray, asp: float):
    left = np.abs(boids[:, 0])
    right = np.abs(asp - boids[:, 0])
    bottom = np.abs(boids[:, 1])
    top = np.abs(1 - boids[:, 1])
    ax = 1. / left**2 - 1. / right**2
    ay = 1. / bottom**2 - 1. / top**2
    boids[:, 4:6] += np.column_stack((ax, ay))


@njit()
def walls(boids: np.ndarray, asp: float):
    c = 1
    x = boids[:, 0]
    y = boids[:, 1]
    a_left = 1 / (np.abs(x) + c)**2
    a_right = -1 / (np.abs(x - asp) + c)**2
    a_bottom = 1 / (np.abs(y) + c)**2
    a_top = -1 / (np.abs(y - 1.) + c)**2
    return np.column_stack((a_left + a_right, a_bottom + a_top))


@njit()
def distance(boids: np.ndarray) -> np.ndarray:
    p = boids[:, :2]
    n = p.shape[0]
    dist = np.zeros(shape=(n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            v = p[i] - p[j]
            d = (v @ v)
            dist[i, j] = d
    dist = np.sqrt(dist)
    return dist


@njit()
def normalize(v):
    v_norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / v_norm


@njit()
def visibility(boids: np.ndarray, perception: float, angle: float) -> np.ndarray:
    vectors = boids[:, :2]
    speeds = boids[:, 2:4]
    n = vectors.shape[0]
    dist = np.zeros(shape=(n, n), dtype=np.float64)
    angles = np.zeros(shape=(n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            v = vectors[i] - vectors[j]
            d = (v @ v)
            dist[i, j] = d
            angles[i, j] = np.dot(normalize(speeds[i]), normalize(v))
    dist = np.sqrt(dist)
    distance_mask = dist < perception
    angle_mask = angles > angle
    mask = np.logical_and(distance_mask, angle_mask)
    np.fill_diagonal(mask, False)
    return mask


@njit()
def cohesion(boids: np.ndarray,
             idx: int,
             neigh_mask: np.ndarray,
             perception: float) -> np.ndarray:
    center = median_axis(boids[neigh_mask, :2], axis=0)
    a = (center - boids[idx, :2]) / perception
    return a


@njit()
def separation(boids: np.ndarray,
               idx: int,
               neigh_mask: np.ndarray) -> np.ndarray:
    d = median_axis(boids[neigh_mask, :2] - boids[idx, :2], axis=0)
    return -d / ((d[0]**2 + d[1]**2) + 1)


@njit()
def alignment(boids: np.ndarray,
              idx: int,
              neigh_mask: np.ndarray,
              v_range: tuple) -> np.ndarray:
    v_mean = median_axis(boids[neigh_mask, 2:4], axis=0)
    a = (v_mean - boids[idx, 2:4]) / (2 * v_range[1])
    return a


@njit()
def noise():
    arr = np.random.rand(2)
    if np.random.rand(1) > .5:
        arr[0] *= -1
    if np.random.rand(1) > .5:
        arr[1] *= -1
    return arr


@njit(parallel=True)
def flocking(boids: np.ndarray,
             perception: float,
             coeffs: np.ndarray,
             asp: float,
             v_range: tuple) -> None:
    n = boids.shape[0]
    mask = visibility(boids, perception, 0)
    wal = walls(boids, asp)
    for i in prange(n):
        if not np.any(mask[i]):
            coh = np.zeros(2)
            alg = np.zeros(2)
            sep = np.zeros(2)
            ns = np.zeros(2)
        else:
            coh = cohesion(boids, i, mask[i], perception)
            alg = alignment(boids, i, mask[i], v_range)
            sep = separation(boids, i, mask[i])
            ns = noise()
        boids[i, 4] = (coeffs[0] * coh[0]
                       + coeffs[1] * alg[0]
                       + coeffs[2] * sep[0]
                       + coeffs[3] * wal[i][0]
                       + coeffs[4] * ns[0])
        boids[i, 5] = (coeffs[0] * coh[1]
                       + coeffs[1] * alg[1]
                       + coeffs[2] * sep[1]
                       + coeffs[3] * wal[i][1]
                       + coeffs[4] * ns[0])


def simulation_step(boids: np.ndarray,
                    asp: float,
                    perception: float,
                    coefficients: np.ndarray,
                    v_range: tuple,
                    dt: float) -> None:
    flocking(boids, perception, coefficients, asp, v_range)
    propagate(boids, dt, v_range)
    periodic_walls(boids, asp)
    wall_avoidance(boids, asp)
