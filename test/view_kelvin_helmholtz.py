# SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

# View Kelvin-Helmholtz instability

import sys

import h5py
import matplotlib.pyplot as plt

print("********************************")
print(" Kelvin-Helmholtz instability")
print("********************************")

filename = sys.argv[1]

with h5py.File(str(filename), "r") as f:
    rho = f["rho"][0, :, :]
    x = f["x_ng"][()]
    y = f["y_ng"][()]
    t = f["current_time"][()]
    iteration = f["iter"][()]

print(f"Final time = {t:.1f} s")
print(f"Iteration number = {iteration}")

xmin = x[0]
xmax = x[-1]
ymin = y[0]
ymax = y[-1]

# ------------------------------------------

plt.figure(figsize=(10, 8), constrained_layout=True)
plt.suptitle("Kelvin-Helmholtz instability")
plt.title(f"Density t = {t:.1f} s")
plt.imshow(rho, cmap="seismic", origin="lower", extent=[xmin, xmax, ymin, ymax])
plt.colorbar(shrink=0.5)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
