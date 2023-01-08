import numpy as np
from numba import njit, prange, objmode, set_num_threads
import time


@njit(parallel=True)
def parameter_izing_model_energy(Lx_array: np.ndarray,
                                 Ly: int,
                                 T_array: np.ndarray,
                                 ):
    """
    Compute average energy of Izing Model system, using Monte-Carlo approach,
    for all parameter configurations, making a grid of all Lx and T values.
    ------------------------
    Description:
    Energy for the grid spin configuration is computed by formula:
    E = sum(- J[i,j] * grid[i,j] * (grid[i+1,j] + grid[i,j+1])
    Mean energy is the expected value of energy for all possible 2^(Lx*Ly)
    configurations of model grid with probabilities:
    P(grid) = e^(-E/kT) / Z, where Z is:
    Z = sum(e^(-E/kT)) - by all E values.
    Mean energies are normed by grid size Lx * Ly.
    After every iterations on Lx parameter the elapsed time in seconds
    is printed in stdout.
    ------------------------
    Param:  Lx_array - 1D array of Lx values for grid size, shape (N, );
            Ly - integer value of Ly for grid size;
            T_array - 1D array of kT values for computations, shape (M, );
    ------------------------
    Return: energies - 2D array of mean normed energies for Izing Models
            shape (N, M)
    ------------------------
    Example:
    >>> Lx_array = np.arange(2, 4, 1).astype(int)
    >>> T_array = np.arange(1, 2.1, 0.1)
    >>> Ly = 2
    >>> parameter_izing_model_energy(Lx_array, Ly, T_array)
    array([[-1.99598209, -1.99170205, -1.98483772, -1.97479989, -1.96114384,
            -1.94360627, -1.92211527, -1.89677817, -1.86785426, -1.83571893,
            -1.80082536],
          [-1.99582582, -1.99124282, -1.98371777, -1.97243779, -1.95670686,
            -1.93602166, -1.91012278, -1.87901623, -1.84296444, -1.80245113,
            -1.75812762]])

    """
    energies = np.zeros((len(Lx_array), len(T_array)))
    for Lx_index in range(len(Lx_array)):
        with objmode(start='f8'):
            start = time.time()
        for T_index in range(len(T_array)):
            Lx = Lx_array[Lx_index]
            T = T_array[T_index]
            N = Lx * Ly
            Z, energy = 0, 0
            for k in prange(2**N):
                E = 0
                spins = np.empty(N, dtype=np.int8)
                flag = np.int64(k)
                for index in range(N):
                    if flag & 1 == 0:
                        spins[index] = -1
                    else:
                        spins[index] = 1
                    flag = flag >> 1
                spins = spins.reshape(Lx, Ly)
                for i in range(-1, Lx-1):
                    for j in range(-1, Ly-1):
                        E -= spins[i, j] * (spins[i, j + 1] + spins[i + 1, j])
                prob = np.exp(- E / T)
                Z += prob
                energy += prob * E
            energies[Lx_index, T_index] = (energy / Z) * (1 / (Lx * Ly))
        with objmode():
             print(time.time() - start, 'seconds elapsed on Lx =', Lx)
    return energies


Lx_array = np.arange(2, 9, 1).astype(int)
T_array = np.arange(1, 5.1, 0.1)
Ly = 4

energies = parameter_izing_model_energy(Lx_array, Ly, T_array).T
np.save('Lx8_izing_energies.npy', energies)