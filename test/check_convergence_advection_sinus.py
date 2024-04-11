import numpy as np
import matplotlib.pyplot as plt
import h5py
import glob
from scipy import stats
import sys

print("********************************")
print(" Convergence advection sinusoide")
print("********************************")

def ExactSolution(x):
    """Exact solution sinusoide density
    input  :
    x      : array : position

    output :
    rho    : array : density
    """
    return 1 + 0.1 * np.sin(2 * np.pi * x)

def Error(filename):
    """Compute L1 error between exact solution and solver resolution
    input    :
    filename : str   : file name solver resolution

    output   :
    error    : float : error value
    """
    with h5py.File(filename, 'r') as f:
        solver = f['rho'][0, 0, :]
        x = f['x'][2:-2] # remove ghost layers
    exact = ExactSolution((x[1:] + x[:-1]) / 2)
    return np.sum(np.abs(exact - solver) * np.diff(x))

if __name__ == "__main__":
    filenames = glob.glob('convergence_test_advection_sinus_[0-9]*.h5')
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
    if(order_error > tol):
        print("FAILURE")
        print(result)
        sys.exit(1)
    else:
        print("SUCCESS")
        print(result)