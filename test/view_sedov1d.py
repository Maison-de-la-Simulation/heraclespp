# View Sedov blast wave 1d

import h5py
import matplotlib.pyplot as plt
import numpy as np
import sys

from exact_sedov import ExactSedov

print("********************************")
print("    Sedov blast wave 1d")
print("********************************")

filename = sys.argv[1]
n = input("Coordinate system: 1 (Cartesian) or 3 (spherical): ")

with h5py.File(str(filename), "r") as f:
    rho = f["rho"][0, 0, :]
    u = f["ux"][0, 0, :]
    P = f["P"][0, 0, :]
    x = f["x_ng"][()]
    t = f["current_time"][()]
    iter = f["iter"][()]
    gamma = f["gamma"][()]
E = 1 / 2 * rho * u**2 + P / (gamma - 1)
xc = (x[:-1] + x[1:]) / 2

# Analytical result ---------------------------------------------------------- #

rho0 = 1
E0 = 1

# n = 1 # Cartesian
# n = 3 # spherical

n = int(n)

r, rho_exact, u_exact, P_exact = ExactSedov(rho0, E0, t, gamma, n)
E_exact = 1 / 2 * rho_exact * u_exact**2 + P_exact / (gamma - 1)

# ------------------------------------------

plt.figure(figsize=(10, 8))
plt.suptitle(f"Sedov blast wave 1d at t = {t:.1f} s")

plt.subplot(221)
plt.plot(r, rho_exact, label="Exact")
plt.plot(xc, rho, label="Solver")
plt.xlabel("x")
plt.ylabel("Density ($kg.m^{-3}$)")
plt.xlim(0, 1)
plt.legend(frameon=False)

plt.subplot(222)
plt.plot(r, u_exact, label="Exact")
plt.plot(xc, u, label="Solver")
plt.xlabel("x")
plt.ylabel("Velocity ($m.s^{-1}$)")
plt.xlim(0, 1)
# plt.ylim(0, 5)
plt.legend(frameon=False)

plt.subplot(223)
plt.plot(r, P_exact, label="Exact")
plt.plot(xc, P, label="Solver")
plt.xlabel("x")
plt.ylabel("Pressure ($kg.m^{-1}.s^{-2}$)")
plt.xlim(0, 1)
plt.legend(frameon=False)

plt.subplot(224)
plt.plot(r, E_exact, label="Exact")
plt.plot(xc, E, label="Solver")
plt.xlabel("x")
plt.ylabel("Volumic energy ($kg.m^{-1}.s^{-2}$)")
plt.xlim(0, 1)
plt.legend(frameon=False)

plt.show()
