#!/usr/bin/env python3

import glob
import h5py
import numpy as np
from scipy import stats
import sys

def ExactSolution(x):
    """Exact solution sinusoide density
    input  :
    x      : array : position

    output :
    rho    : array : density
    """
    return 1 + 0.1 * np.sin(2 * np.pi * x)

def Error(x, rho_simu):
    """Compute L1 error between exact solution and solver simulation
    input    :
    x        : array : position
    rho_simu : array : density on which to compute the error

    output   :
    error    : float : error value
    """
    rho_exact = ExactSolution((x[1:] + x[:-1]) / 2)
    return np.sum(np.abs(rho_exact - rho_simu) * np.diff(x))

def main():
    print("********************************")
    print(" Convergence advection sinusoide")
    print("********************************")

    filenames = glob.glob('convergence_test_advection_sinus_[0-9]*_00000001.h5')
    filenames.sort()

    errors = np.empty(len(filenames))
    points = np.empty(len(filenames))
    for i, filename in enumerate(filenames):
        with h5py.File(filename, 'r') as f:
            errors[i] = Error(f['x_ng'], f['rho'][0, 0, :])
            points[i] = f['nx_glob_ng'][0]

    dx = points[0] / points

    result = stats.linregress(np.log10(dx), np.log10(errors))
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

if __name__ == "__main__":
    main()
