# SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

# Test shock wall 1D
import sys
import h5py
import matplotlib.pyplot as plt

print("********************************")
print("        Shock wall")
print("********************************")

filename = sys.argv[1]

with h5py.File(filename, "r") as f:
    rho = f["rho"][0, 0, :]
    u = f["ux"][0, 0, :]
    P = f["P"][0, 0, :]
    x = f["x_ng"][()]
    t = f["current_time"][()]
    gamma = f["gamma"][()]
e = P / rho / (gamma - 1)
print("Final time =", t, "s")

xc = (x[:-1] + x[1:]) / 2

# ------------------------------------------------------------------------------

plt.figure(figsize=(12, 6))
plt.suptitle(f"Shock wall 1d t = {t:1f} s")
plt.subplot(221)
plt.plot(xc, rho)
plt.xlabel("x")
plt.ylabel("Density ($kg.m^{-3}$)")

plt.subplot(222)
plt.plot(xc, u)
plt.xlabel("x")
plt.ylabel("Velocity ($m.s^{-1}$)")

plt.subplot(223)
plt.plot(xc, P)
plt.xlabel("x")
plt.ylabel("Pressure ($kg.m^{-1}.s^{-2}$)")


plt.subplot(224)
plt.plot(xc, e)
plt.xlabel("x")
plt.ylabel("Internal energy ($m^{2}.s^{-2}$)")

plt.show()
