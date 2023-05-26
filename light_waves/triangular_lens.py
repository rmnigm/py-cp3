import numpy as np
from numba import njit
from utils import clamp, smoothstep


sqrt_3 = 3 ** 0.5
isqrt_3 = 1 / sqrt_3
s_pos = np.array([0, 0.])       # положение источника
s_alpha = - 90.0 * np.pi / 180     # направление источника
s_rot = np.array([
    [np.cos(s_alpha), -np.sin(s_alpha)],
    [np.sin(s_alpha), np.cos(s_alpha)]
])                              # матрица поворота


h = 1.0             # пространственный шаг решетки
c = 1.0             # скорость распространения волн
dt = h / (c * 1.5)  # временной шаг
imp_freq = 400      # "частота" для генерации нескольких волн импульса
imp_sigma = np.array([0.01, 0.03])
n = np.array([       # коэффициент преломления
    1.30,   # R
    1.35,   # G
    1.40    # B
])


@njit
def length(p):
    return np.sqrt((p**2).sum())


@njit
def sd_equilateral_triangle(p, r):
    k = np.sqrt(3.0)
    p[0] = np.abs(p[0]) - r
    p[1] = p[1] + r / k
    if p[0] + k * p[1] > 0.0:
        p = np.array([p[1] - k * p[1], -k * p[0] - p[1]]) / 2.0
    p[0] -= clamp(p[1], -2.0 * r, 0.0)
    return -length(p) * np.sign(p[1])


@njit
def sdf_vesica(p, r, d):
    p = np.abs(p)
    b = np.sqrt(r * r - d * d)
    if (p[1] - b) * d > p[0] * b:
        return length(p - np.array([0.0, b]))
    else:
        return length(p - np.array([-d, 0.0])) - r


@njit
def triangle_mask(nx, ny, a: float = 0.01, b: float = 0.0):
    """
    Расчет треугольной маски размером (ny, nx) с плавным переходом между 0 и 1
    """
    res = np.empty((nx, ny), dtype=np.float64)
    for i in range(nx):
        for j in range(ny):
            uv = np.array([5/6, 0.5]) - np.array([i / ny, j / ny])
            uv = uv - np.array([2.5/6, 0])
            d = sdf_vesica(uv, 0.1, 0.2) - 0.3
            res[i, j] = d
    return res


@njit
def wave_impulse(point: np.ndarray,  # (n, m, 2)
                 pos: np.ndarray,
                 freq: float,        # float
                 sigma: np.ndarray,  # (2, )
                 ):
    d = (point - pos) / sigma
    return np.exp(-0.5 * d @ d) * np.cos(freq * point[0])


@njit
def start_impulse(nx, ny):
    res = np.zeros((nx, ny, 3, 3), dtype=np.float32)
    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            uv = (np.array([j, i]) - 0.5 * np.array([nx, ny])) / ny
            res[i, j, :, :] += wave_impulse(uv, s_pos, imp_freq, imp_sigma)
    return res


def setup(nx, ny, a: float = 0.01, b: float = 0.0):
    mask = triangle_mask(nx, ny, a, b) > 0
    kappa = (c * dt / h) * (mask[None, ...] / n[:, None, None] + (1.0 - mask[None, ...]))
    kappa = kappa.transpose((1, 2, 0))[:, ::-1]
    return {
        'field': start_impulse(nx, ny).astype(np.float32),
        'kappa': kappa.astype(np.float32)
    }