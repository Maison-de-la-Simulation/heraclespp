# SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

# Test the shock tube problem and compare
import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys

print("********************************")
print("    Shock tube spherical 1D")
print("********************************")

file = sys.argv[1]

def read_file(filename):
    with h5py.File(filename, 'r') as f:
        rho = f['rho'][0, 0, :]
        u = f['ux'][0, 0, :]
        P = f['P'][0, 0, :]
        x = f['x_ng'][()]
        T = f['T'][0, 0, :]
        t = f['current_time'][()]
        iter = f['iter'][()]
        gamma = f['gamma'][()]
    e = P / rho / (gamma - 1)
    xc = (x[:-1] + x[1:]) / 2

    print(f"Final time = {t:.1f} s")
    print(f"Iteration number = {iter}")

    return rho, u, P, e, xc, x, gamma, t

# ------------------------------------------------------------------------------

rho, u, P, e, xc, x, gamma, t = read_file(file)

# ------------------------------------------------------------------------------

plt.figure(figsize=(12, 6))
plt.suptitle(f'Shock tube t = {t:.1f} s')
plt.subplot(221)
plt.plot(xc, rho)
plt.xlabel('Position')
plt.ylabel('Density ($kg.m^{-3}$)')

plt.subplot(222)
plt.plot(xc, u)
plt.xlabel('Position')
plt.ylabel('Velocity ($m.s^{-1}$)')

plt.subplot(223)
plt.plot(xc, P)
plt.xlabel('Position')
plt.ylabel('Pressure ($kg.m^{-1}.s^{-2}$)')

plt.subplot(224)
plt.plot(xc, e)
plt.xlabel('Position')
plt.ylabel('Internal energy ($m^{2}.s^{-2}$)')

plt.show()
