# SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

# View Rayleigh Taylor instability

import sys

import h5py
import matplotlib.pyplot as plt


def main():
    print("********************************")
    print("   Shock tube spherical 3d")
    print("********************************")

    filename = sys.argv[1]

    with h5py.File(str(filename), "r") as f:
        rho_1d = f["rho"][0, 0, :]  # rho(r)
        rho2 = f["rho"][:, 0, :]  # rho(r, phi)
        u_1d = f["ux"][0, 0, :]
        P_1d = f["P"][0, 0, :]
        x = f["x_ng"][()]
        y = f["y_ng"][()]
        z = f["z_ng"][()]
        t = f["current_time"][()]
        gamma = f["gamma"][()]

    print(f"Final time = {t:.1f} s")

    e_1d = P_1d / rho_1d / (gamma - 1)

    rmin = x[0]
    rmax = x[-1]
    phi_min = z[0]
    phi_max = z[-1]

    rc = (x[:-1] + x[1:]) / 2

    # ------------------------------------------

    plt.figure(figsize=(10, 8))
    plt.suptitle("Shock tube spherical 3D")
    plt.title(f"Density t = {t:.1f} s")
    plt.imshow(rho2, cmap="seismic", origin="lower", extent=[rmin, rmax, phi_min, phi_max])
    plt.xlabel("Radius (m)")
    plt.ylabel(r"$\phi$ angle (rad)")
    plt.colorbar()

    plt.figure(figsize=(10, 8))
    plt.suptitle(f"Shock tube t = {t:.1f} s")
    plt.subplot(221)
    plt.plot(rc, rho_1d)
    plt.xlabel("Radius (m)")
    plt.ylabel(r"Density ($kg.m^{-3}$)")

    plt.subplot(222)
    plt.plot(rc, u_1d)
    plt.xlabel("Radius (m)")
    plt.ylabel(r"Velocity ($m.s^{-1}$)")

    plt.subplot(223)
    plt.plot(rc, P_1d)
    plt.xlabel("Radius (m)")
    plt.ylabel(r"Pressure ($kg.m^{-1}.s^{-2}$)")

    plt.subplot(224)
    plt.plot(rc, e_1d)
    plt.xlabel("Radius (m)")
    plt.ylabel(r"Internal energy ($m^{2}.s^{-2}$)")

    plt.show()


if __name__ == "__main__":
    main()
