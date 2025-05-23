# SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

# Test the advection sinusoide and compare

import sys
import h5py
import matplotlib.pyplot as plt
import numpy as np

print("********************************")
print("  Advection test : sinus")
print("********************************")

filename = sys.argv[1]

with h5py.File(str(filename), "r") as f:
    rho = f["rho"][0, 0, :]
    x = f["x_ng"][()]
    t = f["current_time"][()]
    iteration = f["iter"][()]

print(f"Final time = {t:.1f} s")
print(f"Iteration number = {iteration}")

xc = (x[:-1] + x[1:]) / 2

# Analytical result ------------------------

rho0 = 1 + 0.1 * np.sin(2 * np.pi * xc)

# ------------------------------------------

plt.figure(figsize=(10, 8))
plt.title("Sinus advection test")
plt.plot(xc, rho0, "--", label="t = 0")
plt.plot(xc, rho, label=f"t = {t:.1f}")
plt.xlabel("Position")
plt.ylabel("Density")
plt.legend(frameon=False)
plt.show()
