# SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

# View Gresho Vortex problem

import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py


def main():
    print("********************************")
    print("        Gresho Vortex")
    print("********************************")

    filename = sys.argv[1]

    with h5py.File(str(filename), "r") as f:
        u_x = f["ux"][0, :, :]
        u_y = f["uy"][0, :, :]
        t = f["current_time"][()]

    print(f"Final time = {t:.1f} s")

    u2d = np.sqrt(u_x**2 + u_y**2)

    # ---------------------------------------------------------------------------- #

    plt.figure(figsize=(15, 8))
    plt.title(f"Gresho Vortex, t = {t:.1e} s")
    plt.imshow(u2d, origin="lower", extent=[-1, 1, -1, 1])
    plt.colorbar(label="$u$")
    plt.plasma()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


if __name__ == "__main__":
    main()
