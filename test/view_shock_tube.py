# SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

# Test the shock tube problem and compare
import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py

from exact_shock_tube import CI, ExactShockTube

print("********************************")
print("    Shock tube 1D Cartesian")
print("********************************")

file = sys.argv[1]


def read_file(filename):
    with h5py.File(filename, "r") as f:
        rho = f["rho"][0, 0, :]
        u = f["ux"][0, 0, :]
        P = f["P"][0, 0, :]
        x = f["x_ng"][()]
        t = f["current_time"][()]
        iteration = f["iter"][()]
        gamma = f["gamma"][()]
    e = P / rho / (gamma - 1)
    xc = (x[:-1] + x[1:]) / 2

    print(f"Final time = {t:.1f} s")
    print(f"Iteration number = {iteration}")

    return rho, u, P, e, xc, x, gamma, t


def analytical_result(x, gamma, t):
    inter = 0.5  # Interface position

    # Left
    rho0l = 1
    u0l = 0
    P0l = 1
    c0l = np.sqrt(gamma * P0l / rho0l)
    var0l = np.array([rho0l, u0l, P0l, c0l])

    # Right
    rho0r = 0.125
    u0r = 0
    P0r = 0.1
    c0r = np.sqrt(gamma * P0r / rho0r)
    var0r = np.array([rho0r, u0r, P0r, c0r])

    Ncell = 1_000
    nodes_exact = np.linspace(x[0], x[-1], Ncell + 1)
    x_exact = (nodes_exact[:-1] + nodes_exact[1:]) / 2

    rho0, u0, P0 = CI(x_exact, inter, var0l, var0r)
    e0 = P0 / rho0 / (gamma - 1)

    rho_exact, u_exact, P_exact, e_exact = ExactShockTube(x_exact, inter, var0l, var0r, t, gamma)

    return rho0, u0, P0, e0, rho_exact, u_exact, P_exact, e_exact, x_exact


# ------------------------------------------------------------------------------

rho, u, P, e, xc, x, gamma, t = read_file(file)
rho0, u0, P0, e0, rho_exact, u_exact, P_exact, e_exact, x_exact = analytical_result(x, gamma, t)

# ------------------------------------------------------------------------------

plt.figure(figsize=(12, 6))
plt.suptitle(f"Shock tube t = {t:.1f} s")
plt.subplot(221)
plt.plot(x_exact, rho0, "--", label="$t_0$")
plt.plot(x_exact, rho_exact, label="Exact")
plt.plot(xc, rho, label="Solver")
plt.xlabel("Position")
plt.ylabel("Density ($kg.m^{-3}$)")
plt.legend(frameon=False)

plt.subplot(222)
plt.plot(x_exact, u0, "--", label="$t_0$")
plt.plot(x_exact, u_exact, label="Exact")
plt.plot(xc, u, label="Solver")
plt.xlabel("Position")
plt.ylabel("Velocity ($m.s^{-1}$)")
plt.legend(frameon=False)

plt.subplot(223)
plt.plot(x_exact, P0, "--", label="$t_0$")
plt.plot(x_exact, P_exact, label="Exact")
plt.plot(xc, P, label="Solver")
plt.xlabel("Position")
plt.ylabel("Pressure ($kg.m^{-1}.s^{-2}$)")
plt.legend(frameon=False)

plt.subplot(224)
plt.plot(x_exact, e0, "--", label="$t_0$")
plt.plot(x_exact, e_exact, label="Exact")
plt.plot(xc, e, label="Solver")
plt.xlabel("Position")
plt.ylabel("Internal energy ($m^{2}.s^{-2}$)")
plt.legend(frameon=False)

plt.show()
