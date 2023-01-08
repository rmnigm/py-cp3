import numpy as np
import numba as nb
from scipy import signal

import matplotlib.pyplot as plt
import seaborn as sns
import holoviews as hv

from tqdm import trange, tqdm
from time import perf_counter
from contextlib import contextmanager

@contextmanager
def catchtime() -> float:
    """
    Measures elapsed time for code executed inside the context manager.
    param: None
    returns: float, code execution time
    """
    start = perf_counter()
    yield lambda: perf_counter() - start
    

L = 100
N_opt, dt_opt = zip((100,0.05),(300,0.02),(700,0.005),(2000,0.0005))

D_default = 1
D_center = 1e-12
dt, N = dt_opt[0], N_opt[0]


@nb.njit(parallel = True)
def diffuse_step_fast(ti, T, D, N, dt, dx):
    ti = ti % 2
    tj = (ti + 1) % 2
    for i in nb.prange(1, N - 1):
        for j in range(1, N - 1):       
            g = dt * D[i, j] / dx**2
            T[tj, i, j] = T[ti, i, j] + \
                     g * (T[ti, i+1, j] + 
                          T[ti, i-1, j] + 
                          T[ti, i, j+1] + 
                          T[ti, i, j-1] -
                          4 * T[ti, i, j])

class PowerSource:
    def __init__(self, x_c, y_c, x_s, y_s, f, dx):
        self.position = np.array([x_c / dx, y_c / dx], dtype=int)
        self.size = np.array([x_s / dx, y_s / dx], dtype=int)
        self.power = f


class HeatTransferModel:
    def __init__(self, L, N, D_default, dt):
        self.L = L
        self.N = N
        self.dx = L / N
        self.D = np.full((N, N), D_default)
        self.T = np.zeros((2, N, N), dtype=np.float32)
        self.dt = dt
        self.power_sources = []
    
    def isolation(self):
        self.T[:,  0, :] = self.T[:,  1, :]
        self.T[:, -1, :] = self.T[:, -2, :]
        self.T[:, :,  0] = self.T[:, :,  1]
        self.T[:, :, -1] = self.T[:, :, -2]
    
    def add_power_source(self, x_c, y_c, x_s, y_s, f):
        s = PowerSource(x_c, y_c, x_s, y_s, f, self.dx)
        self.power_sources.append(s)
    
    def source_step(self, it):
        cur = it % 2
        for s in self.power_sources:
            left = s.position[1] - s.size[1]
            right = left + 2 * s.size[1]
            bottom = s.position[0] - s.size[0]
            top = bottom + 2 * s.size[0]
            self.T[cur, bottom:top, left:right] += self.dt * s.power(it * self.dt)
    
    def step(self, it):
        self.source_step(it)
        diffuse_step_fast(it, self.T, self.D, self.N, self.dt, self.dx)
        self.isolation()
        return self.T[(it + 1) % 2]

def power_first(dt):
    p, teta, phi = 100, 50, 0
    return p * (np.sin(2 * np.pi * dt / teta + phi) + 1)

def power_second(dt):
    p, teta, phi = 100, 77, 10
    return p * (np.sin(2 * np.pi * dt / teta + phi) + 1)

opts = zip(N_opt, dt_opt)


for index, (N, dt) in enumerate(opts):
    model = HeatTransferModel(100, N, D_default, dt)

    model.add_power_source(0.2*L, 0.5*L, L/25, 0.2*L, power_first)
    model.add_power_source(0.5*L, 0.2*L, 0.2*L, L/25, power_second)

    model.T[:,  0,  :] = 0.0
    model.T[:, -1,  :] = 50.0
    model.T[:,  :,  0] = 50.0
    model.T[:,  :, -1] = 0.0

    probes = {0: (int((0.3*L) // model.dx), int((0.3*L) // model.dx)),
              1: (int((0.3*L) // model.dx), int((0.7*L) // model.dx)),
              2: (int((0.7*L) // model.dx), int((0.7*L) // model.dx)),
              3: (int((0.7*L) // model.dx), int((0.3*L) // model.dx))}

    # задаем круг в центре с плохой проводимостью тепла

    R = (L / 5) // model.dx
    for i in range(model.N):
        for j in range(model.N):
            if (i-N//2)**2 + (j-N//2)**2 <= R**2:
                model.D[i, j] = D_center

    max_time = 600.0
    iterations = int(max_time / dt)

    probe_values = np.empty((4, iterations))
    
    with catchtime() as t:
        for i in trange(iterations):
            T = model.step(i)
            for j, (x, y) in probes.items():
                probe_values[j, i] = T[x, y]
    print(f"Execution time on {(N, dt)}: {t():.4f} secs")

    np.save('T_'+str(index)+'.npy', T)
    np.save('Probe_coord_'+str(index)+'.npy', np.array([list(x) for x in probes.values()]))
    np.save('Prob_'+str(index)+'.npy', probe_values)