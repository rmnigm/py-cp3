import numpy as np
from numba import njit


@njit
def rot(x):
    """
    Rotation 2D matrix by angle x
    :param x: angle of rotation in radians
    :return: np.array, (2, 2)
    """
    return np.array([[np.cos(x), -np.sin(x)], [np.sin(x), np.cos(x)]])


h = 1.0  # пространственный шаг решетки
c = 1.0  # скорость распространения волн
dt = h / (c * 1.5)  # временной шаг
imp_freq = 400  # "частота" для генерации нескольких волн импульса
imp_sigma = np.array([0.005, 0.025])
n = np.array([  # коэффициент преломления
    1.30,  # R
    1.50,  # G
    1.70   # B
])


@njit
def length(p):
    """
    Calculate euclidean norm of vector p
    :param p: np.array, shape (m, )
    :return: float, vector norm
    """
    return np.sqrt((p ** 2).sum())


@njit
def sdf_parabola(pos, k):
    """Signed distance function for parabola half-planes in 2D"""
    pos[0] = abs(pos[0])
    ik = 1.0 / k
    p = ik * (pos[1] - 0.5 * ik) / 3.0
    q = 0.25 * ik * ik * pos[0]
    h = q * q - p * p * p
    r = np.sqrt(abs(h))
    if h > 0:
        x = pow(q + r, 1.0 / 3.0) + pow(abs(q - r), 1.0 / 3.0) * np.sign(p)
    else:
        x = 2.0 * np.cos(np.arctan(r / q) / 3.0) * np.sqrt(p)
    return length(pos - np.array([x, k * x * x])) * np.sign(pos[0] - x)


@njit
def sdf_hyberbola(p, k, he):
    """Signed distance function for hyperbola half-planes in 2D"""
    p = np.abs(p)
    p = np.array([p[0] - p[1], p[0] + p[1]]) / np.sqrt(2.)
    x2 = p[0] * p[0] / 16.
    y2 = p[1] * p[1] / 16.
    r = k * (4.0 * k - p[0] * p[1] ) / 12.
    q = (x2 - y2) * k**2
    h = q**2 + r**3
    if h < 0:
        m = np.sqrt(-r)
        u = m * np.cos(np.arccos( q / (r * m)) / 3)
    else:
        m = pow(np.sqrt(h) - q, 1 / 3.)
        u = (m - r/m) / 2.
    w = np.sqrt(u + x2)
    b = k * p[1] - x2 * p[0] * 2.
    t = p[0] / 4. - w + np.sqrt(2. * x2 - u + b / w / 4.)
    t = max(t, np.sqrt(he * he * 0.5 + k) - he / np.sqrt(2.))
    d = length(p - np.array([t, k/t]))
    return d if p[0] * p[1] < k else -d


@njit
def mask(nx, ny):
    """
    Calculation of mask array for two symmetrical lenses.
    Lenses have convex parabolic and concave hyperbolic sides.
    :param nx: mask width by
    :param ny: mask height by Y axis
    :return: matrix
    """
    res = np.empty((nx, ny), dtype=np.float64)
    for i in range(nx):
        for j in range(ny):
            uv = np.array([i / ny, j / ny])
            uv_hyperbola = uv - np.array([5 / 6, 0.5])
            uv_hyperbola *= 7
            uv_parabola = np.array([5 / 6, 0.5]) - uv + np.array([0.1, 0])
            d1 = sdf_parabola(rot(3 * np.pi / 2) @ uv_parabola, 1)
            d2 = sdf_hyberbola(uv_hyperbola, 0.7, 0.8)
            first_lens = (d1 < 0 and d2 > 0
                          and 0.3 > uv_parabola[1] > -0.3
                          and uv_parabola[0] < 0
                          )
            res[i, j] = int(first_lens)
            uv_parabola[0] *= -1
            uv_parabola += np.array([0.2, 0])
            d1 = sdf_parabola(rot(3 * np.pi / 2) @ uv_parabola, 1)
            d2 = sdf_hyberbola(uv_hyperbola, 0.7, 0.8)
            second_lens = (d1 < 0 and d2 > 0
                           and 0.3 > uv_parabola[1] > -0.3
                           and uv_parabola[0] < 0
                           )
            res[i, j] = int(first_lens or second_lens)
    return res


@njit
def wave_impulse(point: np.ndarray,  # (n, m, 2)
                 pos: np.ndarray,
                 freq: float,  # float
                 sigma: np.ndarray,  # (2, )
                 ):
    """Calculate impulse power in a point by source position and frequency"""
    d = (point - pos) / sigma
    return np.exp(-0.5 * d @ d) * np.cos(freq * point[0])


@njit
def start_impulse(nx, ny):
    """Create an angled impulse light wave for matrix of points"""
    # constants
    s_pos = np.array([-0.6, 0])  # положение источника
    s_alpha = -np.radians(10.)  # направление источника
    
    res = np.zeros((nx, ny, 3), dtype=np.float32)
    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            uv = np.array([i / ny, j / ny]) - np.array([5 / 6, 0.5])
            uv = rot(s_alpha) @ uv
            res[i, j, :] += wave_impulse(uv, s_pos, imp_freq, imp_sigma)
    return res


def setup(nx, ny, a: float = 0.01, b: float = 0.0):
    """Setup the two-lens model with angled impulse for numerical modelling"""
    res_mask = mask(nx, ny)
    kappa = (c * dt / h) * (res_mask[None, ...] / n[:, None, None] + (1.0 - res_mask[None, ...]))
    kappa = kappa.transpose((1, 2, 0))[:, ::-1]
    return {
        'field': start_impulse(nx, ny).astype(np.float32),
        'kappa': kappa.astype(np.float32)
    }
