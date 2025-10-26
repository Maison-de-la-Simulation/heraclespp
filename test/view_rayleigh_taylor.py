# SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

# View Rayleigh Taylor instability

import sys

import h5py
import matplotlib.pyplot as plt


def main():
    print("********************************")
    print("Rayleigh Taylor instability 2d")
    print("********************************")

    filename = sys.argv[1]

    with h5py.File(str(filename), "r") as f:
        rho = f["rho"][0, :, :]
        u = f["ux"][0, :, :]
        P = f["P"][0, :, :]
        Py = f["P"][0, :, 0]
        x = f["x_ng"][()]
        y = f["y_ng"][()]
        fx = f["fx0"][0, :, :]
        t = f["current_time"][()]
        iteration = f["iter"][()]

    print(f"Final time = {t:.1f} s")
    print(f"Iteration number = {iteration}")

    xmin = x[0]
    xmax = x[-1]
    ymin = y[0]
    ymax = y[-1]

    # ------------------------------------------

    plt.figure(figsize=(10, 8))
    plt.suptitle("Rayleigh Taylor instability")
    plt.title(f"Density t = {t:.1f} s")
    plt.imshow(rho, cmap="inferno", origin="lower", extent=[xmin, xmax, ymin, ymax])
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")

    plt.figure(figsize=(10, 8))
    plt.title("Passive scalar")
    plt.imshow(fx, cmap="inferno", origin="lower", extent=[xmin, xmax, ymin, ymax])
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


if __name__ == "__main__":
    main()
