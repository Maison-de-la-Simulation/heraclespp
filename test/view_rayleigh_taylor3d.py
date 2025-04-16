# SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

# View Rayleigh Taylor instability

import sys

import h5py
import matplotlib.pyplot as plt

print("********************************")
print("Rayleigh Taylor instability 3d")
print("********************************")

filename = sys.argv[1]

with h5py.File(str(filename), "r") as f:
    # print(f.keys())
    fx = f["fx"][0, :, 0, :]
    rho = f["rho"][:, 0, :]
    x = f["x_ng"][()]
    z = f["z_ng"][()]
    t = f["current_time"][()]
    iter = f["iter"][()]

print(f"Final time = {t:.1f} s")
print(f"Iteration number = {iter}")

xmin = x[0]
xmax = x[-1]
zmin = z[0]
zmax = z[-1]

# ------------------------------------------

plt.figure(figsize=(10, 8))
plt.suptitle("Rayleigh Taylor instability")
plt.title(f"Density t = {t:.1f} s")
plt.imshow(rho, cmap="seismic", origin="lower", extent=[xmin, xmax, zmin, zmax])
plt.colorbar()
plt.xlabel("x")
plt.ylabel("z")

plt.figure(figsize=(10, 8))
plt.suptitle("Rayleigh Taylor instability")
plt.title(f"Passive scalar t = {t:.1f} s")
plt.imshow(fx, cmap="seismic", origin="lower", extent=[xmin, xmax, zmin, zmax])
plt.colorbar()
plt.xlabel("x")
plt.ylabel("z")
plt.show()
