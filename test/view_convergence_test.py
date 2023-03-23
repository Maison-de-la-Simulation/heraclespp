import numpy as np
import matplotlib.pyplot as plt
import h5py

from param import *

print("********************************")
print(" Convergence test")
print("********************************")

def ExactSolution(N):
    """Exact solution sinusoide density
    input  :
    N      : float : size array

    output :
    rho    : array : density
    """
    dx = L / N
    x = np.zeros(N) # Left interface
    for i in range(N):
        x[i] = i * dx
    rho = np.zeros(N)
    for i in range(N):
        rho[i] = 1 * np.exp(-15 * ((1. / 2)  - x[i])**2)
    return rho

def Error(filename):
    """Compute error between exact solution and solver resolution
    input    :
    filename : str   : file name solver resolution

    output   :
    error    : float : error value
    """
    with h5py.File(filename, 'r') as f : 
        solver = f['rho'][:, 0, 0]
    N = len(solver)
    exact = ExactSolution(N)
    return np.sum((exact - solver)**2) * (1 / N)

#err100 = Error('example/ad_sin.h5')
err200 = Error('example/ad_sin_200p.h5')
err300 = Error('example/ad_sin_300p.h5')
err400 = Error('example/ad_sin_400p.h5')
err500 = Error('example/ad_sin_500p.h5')
err600 = Error('example/ad_sin_600p.h5')
err700 = Error('example/ad_sin_700p.h5')
err800 = Error('example/ad_sin_800p.h5')
err900 = Error('example/ad_sin_900p.h5')
err1000 = Error('example/ad_sin_1000p.h5')
val_error = np.array([err200, err300, err400, err500, err600, err700, err800, err900, err1000])

points = np.array([200, 300, 400, 500, 600, 700, 800, 900, 1_000])
dx = 1 / points

plt.figure(figsize=(10,8))
plt.plot(points, val_error , 'o-', label='Ordre 2')
plt.plot(points, dx**2, 'o-', label='$dx^{2}$ error')
plt.xticks([100, 200, 300, 400, 500, 600, 700, 800, 900, 1_000])
plt.yscale('log')
plt.ylabel('Error'); plt.xlabel('Number of points')
plt.grid()
plt.legend()
plt.show()