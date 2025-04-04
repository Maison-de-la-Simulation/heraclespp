# SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

# View Rayleigh Taylor instability

import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np

from exact_sedov import ExactSedov

print("********************************")
print("   Sedov blast wave 3d")
print("********************************")

filename = sys.argv[1]

with h5py.File(filename, 'r') as f:
    P_3d = f['P'][:, :, :] # P(phi, theta, r)

nr = P_3d.shape[2]
nth = P_3d.shape[1]
nph = P_3d.shape[0]
print("Size = ", nr, nth, nph)

ind_th = int(nth / 2)
ind_ph = int(nph / 2)

with h5py.File(str(filename), 'r') as f:
    #print(f.keys())
    rho_1d = f['rho'][ind_ph, ind_th, :] # rho(r)
    u_1d = f['ux'][ind_ph, ind_th, :] # u(r)
    P_1d = f['P'][ind_ph, ind_th, :] # P(r)
    rho_r_th = f['rho'][ind_ph, :, :] # rho(r, theta)
    rho_r_ph = f['rho'][:, ind_th, :] # rho(r, phi)
    x = f['x_ng'][()]
    y = f['y_ng'][()]
    z = f['z_ng'][()]
    t = f['current_time'][()]
    iter = f['iter'][()]
    gamma = f['gamma'][()]

print(f"Final time = {t:.5f} s")
print(f"Iteration number = {iter}")
print("Shape = ", P_3d.shape)

E = 1 / 2 * rho_1d * u_1d**2 + P_1d / (gamma - 1)

# Grid
r = x[:]
dr = r[1] - r[0]
rc = (r[:-1] + r[1:]) / 2
th = y[:]
thc = (th[:-1] + th[1:]) / 2
ph = z[:]
phc = (ph[:-1] + ph[1:]) / 2

# Analytical result 1d ------------------------

rho0 = 1
E0 = 1

n = 3 # spherical

r_exact, rho_exact, u_exact, P_exact = ExactSedov(rho0, E0, t, gamma, n)
E_exact = 1 / 2 * rho_exact * u_exact**2 + P_exact / (gamma - 1)

rc_exact = r_exact+1 + dr / 2

# Analytical result 2d ------------------------

beta = 1.15 #gamma = 5/3

r_choc = beta * (E0 * t**2 / rho0)**(1/5)

theta = np.linspace(0, 2*np.pi, len(x))
x_choc = r_choc * np.sin(theta)
z_choc = r_choc * np.cos(theta) + 1

# Spherical to cartesian -------------------

R, Theta = np.meshgrid(rc, thc)

X = R * np.sin(Theta)
Z = R * np.cos(Theta)

plt.figure(figsize=(10, 8))
plt.suptitle('Sedov blast wave spherical 3d in Cartesian coordinates')
plt.plot(x_choc, z_choc, '--', c='black')
plt.axvline(x=0, color='black', linestyle='--')
plt.contourf(Z, X, rho_r_th, cmap='gnuplot')
plt.gca().set_aspect('equal')
plt.xlabel('x')
plt.ylabel('z')
plt.colorbar(shrink=0.55)
plt.show()

# Distance from the explosion -------------------

# explosion coordinates
x0 = 1 * np.sin(np.pi / 2) * np.cos(np.pi / 2)
y0 = 1 * np.sin(np.pi / 2) * np.sin(np.pi / 2)
z0 = 1 * np.cos(np.pi / 2)

rdist = np.zeros(P_3d.shape)

""" for i in range(nr):
    print(i)
    for j in range(nth):
    #j = int(nth / 2)
        for k in range(nphi):
        #k = int(nphi / 2)
            xcart = r[i] * np.sin(th[j]) * np.cos(phi[k])
            ycart = r[i] * np.sin(th[j]) * np.sin(phi[k])
            zcart = r[i] * np.cos(th[j])
            rdist[k, j, i] = np.sqrt((xcart - x0)**2 + (ycart - y0)**2 + (zcart - z0)**2)
print("tableau done")

plt.figure(figsize=(10,8))
for i in range(nr):
    print(i)
    for j in range(nth):
        for k in range(nphi):
        #j = int(nth / 2)
        #k = int(nphi / 2)
            plt.plot(rdist[k, j, i], P_3d[k, j, i], "o", color='#1f77b4')
plt.plot(r_exact, P_exact, "--", label=f"Exact", color='black')
plt.xlabel("Radius (m)"); plt.ylabel('Pressure ($kg.m^{-1}.s^{-2}$)')
plt.legend()
plt.xlim(0, 1)
plt.show() """

# ------------------------------------------

plt.figure(figsize=(8,8))
plt.subplot(221)
plt.plot(rc, rho_1d, label="Solver")
plt.plot(rc_exact, rho_exact, label="Exact")
plt.xlabel("Radius (m)")
plt.ylabel(r"Density")
plt.xlim(x[0], x[len(x)-1])
plt.grid()
plt.legend()

plt.subplot(222)
plt.plot(rc, E, label="Solver")
plt.plot(rc_exact, E_exact, label="Exact")
plt.xlabel("Radius (m)")
plt.ylabel('Volumic energy ($kg.m^{-1}.s^{-2}$)')
plt.xlim(x[0], x[len(x)-1])
plt.grid()
plt.legend()

plt.subplot(223)
plt.plot(rc, P_1d, label="Solver")
plt.plot(rc_exact, P_exact, label="Exact")
plt.xlabel("Radius (m)")
plt.ylabel('Pressure ($kg.m^{-1}.s^{-2}$)')
plt.xlim(x[0], x[len(x)-1])
plt.grid()
plt.legend()
plt.show()
