#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

import argparse
import sys
import typing

import h5py
import numpy as np
from scipy import stats


def exact_solution(x: np.ndarray) -> np.ndarray:
    """Exact solution sinusoide density
    input  :
    x      : array : position

    output :
    rho    : array : density
    """
    return 1 + 0.1 * np.sin(2 * np.pi * x)


def error(x: np.ndarray, rho_simu: np.ndarray):
    """Compute L1 error between exact solution and solver simulation
    input    :
    x        : array : position
    rho_simu : array : density on which to compute the error

    output   :
    error    : float : error value
    """
    rho_exact = exact_solution((x[1:] + x[:-1]) / 2)
    return np.sum(np.abs(rho_exact - rho_simu) * np.diff(x))


def check_convergence_order(filenames: typing.List[str]):
    """Check convergence order from the list of the given h5 files"""

    def key(filename: str):
        with h5py.File(filename) as f:
            return f["nx_glob_ng"][0]

    filenames.sort(key=key)

    errors = np.empty(len(filenames))
    points = np.empty(len(filenames))
    for i, filename in enumerate(filenames):
        with h5py.File(filename) as f:
            errors[i] = error(f["x_ng"], f["rho"][0, 0, :])
            points[i] = f["nx_glob_ng"][0]

    dx = points[0] / points

    result = stats.linregress(np.log10(dx), np.log10(errors))
    theoretical_slope = 2
    tol = 0.05
    order_error = np.fabs(result.slope - theoretical_slope) / theoretical_slope
    if np.isfinite(order_error) and order_error < tol:
        print("SUCCESS")
        print(result)
    else:
        print("FAILURE")
        print(result)
        sys.exit(1)


if __name__ == "__main__":

    def main():
        """main function"""
        parser = argparse.ArgumentParser(description="")
        parser.add_argument("filenames", nargs="+", type=str, help="")
        args = parser.parse_args()

        print("********************************")
        print(" Convergence advection sinusoide")
        print("********************************")

        check_convergence_order(args.filenames)

    main()
