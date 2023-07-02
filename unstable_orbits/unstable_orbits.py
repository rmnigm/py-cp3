import numpy as np
import numba as nb
import matplotlib.pyplot as plt

# cache dict for JIT compilation
jkw = dict(cache=True)


# runge-kutta method constants
a = np.array([
    [0, 0, 0, 0, 0, 0],
    [1/5, 0, 0, 0, 0, 0],
    [3/40, 9/40, 0, 0, 0, 0],
    [3/10, -9/10, 6/5, 0, 0, 0],
    [226/729, -25/27, 880/729, 55/729, 0, 0],
    [-181/270, 5/2, -266/297, -91/27, 189/55, 0]
])
c = np.array([0, 1/5, 3/10, 3/5, 2/3, 1])
b = np.array([19/216, 0, 1000/2079, -125/216, 81/88, 5/56])


# integrator, system of ODEs and bisect method for speed and plane search
@nb.cfunc('f8[:](f8, f8[:], f8[:])', **jkw)
def crtbp_ode(t, s, mc):
    """Calculate the left part of the CRTBP system of ODEs by coordinates and speed"""
    mu2 = mc[0]
    mu1 = 1 - mu2

    x, y, z, vx, vy, vz = s[:6]

    yz2 = y * y + z * z
    r13 = ((x + mu2) * (x + mu2) + yz2) ** (-1.5)
    r23 = ((x - mu1) * (x - mu1) + yz2) ** (-1.5)

    mu12r12 = (mu1 * r13 + mu2 * r23)

    ax = 2 * vy + x - (mu1 * (x + mu2) * r13 + mu2 * (x - mu1) * r23)
    ay = -2 * vx + y - mu12r12 * y
    az = - mu12r12 * z

    out = np.array([vx, vy, vz, ax, ay, az])
    return out


@nb.njit(**jkw)
def rk_step(f, t, s, h, mc):
    """Make one step of general Runge-Kutta integration algorithm with use of global-variable constants"""
    ss = 6
    k = []
    k.append(f(t, s, mc))
    for i in range(1, ss):
        diff = s
        for j in range(i):
            diff += h * a[i, j] * k[j]
        k.append(f(t + c[i] * h, diff, mc))
    diff = s
    for i in range(ss):
        diff += h * b[i] * k[i]
    return diff


@nb.njit(**jkw)
def rk_nsteps(f, t, s, h, mc, n):
    """Make n consecutive steps in time of integration algorithm"""
    arr = np.empty((n + 1, s.shape[0] + 1))
    arr[:, 0] = t + h * np.arange(n + 1)
    arr[0, 1:] = s

    for i in range(n):
        arr[i + 1, 1:] = rk_step(f,           # right part of SODE
                                 arr[i, 0],   # t_0
                                 arr[i, 1:],  # s_0
                                 h,           # time step dt
                                 mc)          # model params
    return arr


@nb.njit(**jkw)
def rk_nsteps_plane(f, t, s, h, mc, n, pl):
    """
    Make N consecutive steps in time of integration algorithm before point reaches
    one of two planes, defined in parameters.
    """
    arr = np.empty((n + 1, s.shape[0] + 1))
    arr[:, 0] = t + h * np.arange(n + 1)
    arr[0, 1:] = s

    for i in range(n):
        arr[i + 1, 1:] = rk_step(f,           # правая часть СОДУ
                                 arr[i, 0],   # t_0
                                 arr[i, 1:],  # s_0
                                 h,           # шаг dt
                                 mc)          # параметры модели
        x = arr[i + 1, 1]
        if x < pl[0] or x > pl[1]:
            break
    return arr[:i + 2]


@nb.njit(**jkw)
def get_plane(vy, f, s, h, mc, n, pl):
    """Calculate whether point with defined start params intercepts left or right plane"""
    s0 = s.copy()
    s0[4] = vy
    arr = rk_nsteps_plane(f, 0., s0, h, mc, n, pl)
    x = arr[-1, 1]
    mid = pl.mean()
    return -1 if x < mid else 1


@nb.njit(**jkw)
def bisect(f, a, b, args, xtol, maxiter=100):
    """General bisection algorithm, search for root of function"""
    xa = a * 1.
    xb = b * 1.
    fa = f(xa, *args)
    for itr in range(maxiter):
        xm = (xa + xb) / 2
        fm = f(xm, *args)
        if fm * fa >= 0:
            xa = xm
        else:
            xb = xm
        if fm == 0 or abs((xa - xb) / 2) < xtol:
            break
    return xm


def preset_grid(xmin, xmax, zmin, zmax, N):
    """Helper function to create a grid of NxN start points with different coordinates"""
    x_step = (xmax - xmin) / N
    z_step = (zmax - zmin) / N
    speeds = np.zeros((N, N, 6))
    xs = np.arange(xmin, xmax, x_step)
    zs = np.arange(zmin, zmax, z_step)
    for i in range(N):
        for j in range(N):
            speeds[i, j][[0, 2]] = xs[i], zs[j]
    return speeds


@nb.njit(**jkw)
def jacobi(s, mc):
    """Calculate Jacobi constant by the parameters for CBTRP SODE"""
    x, y, z, vx, vy, vz = s
    mu2 = mc[0]
    mu1 = 1 - mu2
    r1 = ((x + mu2) ** 2 + y ** 2 + z ** 2) ** (-0.5)
    r2 = ((x - mu1) ** 2 + y ** 2 + z ** 2) ** (-0.5)
    omega = 0.5 * (x ** 2 + y ** 2) + mu1 * r1 + mu2 * r2
    return 2 * omega - vx ** 2 - vy ** 2 - vz ** 2


@nb.njit(parallel=True)
def compute_grid(f, starter_speeds, h, mc, n, pl):
    """
    Compute the initial speed value for point to stay in the libration point
    for all start postions from grid starter_speeds using bisection algorithm
    """
    N = len(starter_speeds)
    speeds = np.zeros((N, N), dtype=np.float64)
    jacobi_constants = np.zeros((N, N), dtype=np.float64)
    for i in nb.prange(N):
        for j in range(N):
            s = starter_speeds[i, j].astype(np.float64)
            speeds[i, j] = bisect(get_plane, -1, 1, args=(f, s, h, mc, n, pl), xtol=1e-12)
            s[4] = speeds[i, j]
            jacobi_constants[i, j] = jacobi(s, mc)
    return speeds, jacobi_constants
