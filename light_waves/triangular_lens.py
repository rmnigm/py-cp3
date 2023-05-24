import numpy as np
from numba import njit
from utils import clamp, smoothstep


sqrt_3 = 3 ** 0.5
isqrt_3 = 1 / sqrt_3
s_pos = np.array([-0.5, 0.])       # положение источника
s_alpha = - 25.0 * np.pi / 180     # направление источника
s_rot = np.array([
    [np.cos(s_alpha), -np.sin(s_alpha)],
    [np.sin(s_alpha), np.cos(s_alpha)]
])                              # матрица поворота
prism_s = 3.0                   # масштаб треугольной призмы
prism_pos = np.array([0., 0])   # положение призмы
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
def sd_equilateral_triangle(p):
    """
    SDF равностороннего треугольника
    :param p:
    :return:
    """
    r = np.array([abs(p[0]) - 1.0, p[1] + isqrt_3])
    if r[0] + sqrt_3 * r[1] > 0.0:
        r = np.array([r[0] - sqrt_3 * r[1], -sqrt_3 * r[0] - r[1]]) * 0.5
    r[0] -= clamp(r[0], -2.0, 0.0)
    return -(r @ r)**0.5 * np.sign(r[1])


@njit
def triangle_mask(nx, ny, a: float = 0.01, b: float = 0.0):
    """
    Расчет треугольной маски размером (ny, nx) с плавным переходом между 0 и 1
    """
    res = np.empty((nx, ny), dtype=np.float64)
    for i in range(nx):
        for j in range(ny):
            uv = 0.5 - np.array([i / nx, j / ny])
            d = sd_equilateral_triangle((uv + prism_pos) * prism_s)
            res[i, j] = smoothstep(a, b, d)
    return res


@njit
def wave_impulse(point: np.ndarray,  # (n,m,2)
                 pos: np.ndarray,
                 freq: float,  # float
                 sigma: np.ndarray,  # (2,)
                 ):
    d = (point - pos) / sigma
    return np.exp(-0.5 * d @ d) * np.cos(freq * point[0])


@njit
def start_impulse(nx, ny):
    res = np.zeros((nx, ny, 3, 3), dtype=np.float32)
    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            uv = (np.array([j, i]) - 0.5 * np.array([nx, ny])) / ny
            res[i, j, :, :] += wave_impulse(s_rot @ uv, s_pos, imp_freq, imp_sigma)
    return res


def setup(nx, ny, a: float = 0.01, b: float = 0.0):
    mask = triangle_mask(nx, ny, a, b)
    kappa = (c * dt / h) * (mask[None, ...] / n[:, None, None] + (1.0 - mask[None, ...])).transpose((1, 2, 0))
    return {
        'field': start_impulse(nx, ny).astype(np.float32),
        'kappa': kappa.astype(np.float32)
    }