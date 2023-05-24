import numpy as np
from numba import njit


@njit
def mix(a: float, b: float, x: float) -> float:
    return a * x + b * (1.0 - x)


@njit
def clamp(x, low, high):
    return np.maximum(np.minimum(x, high), low)


@njit
def smoothstep(edge0, edge1, x):
    n = (x - edge0) / (edge1 - edge0)
    t = clamp(n, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)
