import numpy as np
import matplotlib.pyplot as plt
import h5py
import glob
from scipy import stats
import sys

L = 1

print("********************************")
print(" Convergence advection sinusoide")
print("********************************")

def ExactSolution(N):
    """Exact solution sinusoide density
    input  :
    N      : float : size array

    output :
    rho    : array : density
    """
    dx = L / N
    x = np.zeros(N) # Center
    for i in range(N):
        x[i] = i * dx  + dx / 2
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
    with h5py.File(filename, 'r') as f:
        solver = f['rho'][0, 0, :]
    N = len(solver)
    exact = ExactSolution(N)
    return np.sum((exact - solver)**2) * (1 / N)

if __name__ == "__main__":
    filenames = glob.glob('convergence_test_advection_gaussian_[0-9]*.h5')
    filenames.sort()
    val_error = np.empty(len(filenames))
    for i in range(len(filenames)):
        val_error[i] = Error(filenames[i])

    points = np.empty(len(filenames))
    for i in range(len(filenames)):
        with h5py.File(filenames[i], 'r') as f:
            points[i] = f['nx_glob_ng'][0]

    dx = points[0] / points

    result = stats.linregress(np.log10(dx), np.log10(val_error))
    theoretical_slope = 2
    tol = 0.05
    order_error = np.fabs(result.slope - theoretical_slope) / theoretical_slope
    if(result.slope < 2 - tol):
    #if order_error > tol:
        print("FAILURE")
        print(result)
        sys.exit(1)
    else:
        print("SUCCESS")
        print(result)