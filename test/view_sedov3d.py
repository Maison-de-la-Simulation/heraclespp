# View Rayleigh Taylor instability

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py
import sys

from exact_sedov import ExactSedov

print("********************************")
print("   Sedov blast wave 3d")
print("********************************")

filename = sys.argv[1]

ind_th = 64
ind_ph = 128

with h5py.File(str(filename), 'r') as f :
    #print(f.keys())
    rho_1d = f['rho'][ind_ph, ind_th, :] # rho(r)
    u_1d = f['ux'][ind_ph, ind_th, :] # u(r)
    P_1d = f['P'][ind_ph, ind_th, :] # P(r)
    rho_r_th = f['rho'][ind_ph, :, :] # rho(r, theta)
    rho_r_ph = f['rho'][:, ind_th, :] # rho(r, phi)
    u2 = f['ux'][0, :, :] # u(r, theta)
    P2 = f['P'][0, :, :] # P(r, theta)
    x = f['x'][()]
    y = f['y'][()]
    z = f['z'][()]
    t = f['current_time'][()]
    iter = f['iter'][()]
    gamma = f['gamma'][()]

print(f"Final time = {t:.5f} s")
print(f"Iteration number = {iter}")

E = 1 / 2 * rho_1d * u_1d**2 + P_1d / (gamma - 1)

rmin = x[2]
rmax = x[len(x)-3]
th_min = y[2]
th_max = y[len(y)-3]
phi_min = z[2]
phi_max = z[len(z)-3]

nr = len(rho_1d)
dr = (rmax - rmin) / nr
r = np.zeros(nr)
rc = np.zeros(nr)
for i in range(2, nr+2):
    r[i-2] = x[i]
    rc[i-2] = x[i] + dr / 2

nth = rho_r_th.shape[1]
dth = (th_max - th_min) / nth
th = np.zeros(nth)
for i in range(2, nth+2):
    th[i-2] = y[i]

nphi = rho_r_ph.shape[0]
dphi = (phi_max - phi_min) / nphi
phi = np.zeros(nphi)
for i in range(2, nphi+2):
    phi[i-2] = z[i]

# Analytical result ------------------------

rho0 = 1
E0 = 1

n = 3 # spherical

r_exact, rho_exact, u_exact, P_exact = ExactSedov(rho0, E0, t, gamma, n)
E_exact = 1 / 2 * rho_exact * u_exact**2 + P_exact / (gamma - 1)

rt = r_exact+1

# Spherical to cartesian -------------------

xcart = r * np.sin(th)
zcart = r * np.cos(th)

# explosion corrdinates
x0 = 1 * np.sin(np.pi / 2)
z0 = 1 * np.cos(np.pi / 2)

rdist = np.zeros(len(xcart))
for i in range(len(rdist)):
    rdist[i] = np.sqrt((xcart[i] - x0)**2 + (zcart[i] - z0)**2)

print(r, rdist)

# ------------------------------------------

plt.figure(figsize=(10,8))
plt.suptitle('Rayleigh Taylor spherical 3D')
plt.title(f'Density t = {t:.5f} s')
plt.imshow(rho_r_th, cmap='seismic', origin='lower', extent=[rmin, rmax, th_min, th_max])
plt.xlabel("Radius (m)"); plt.ylabel(r"$\theta$ angle (rad)")
plt.colorbar()

plt.figure(figsize=(8,8))
plt.subplot(221)
plt.plot(r, rho_1d, label="Solver")
plt.plot(rt, rho_exact, label=f"Exact")
plt.xlabel("Radius (m)"); plt.ylabel(r"Density")
plt.xlim(x[0], x[len(x)-1])
plt.grid()
plt.legend()

plt.subplot(222)
plt.plot(r, E, label="Solver")
plt.plot(rt, E_exact, label=f"Exact")
plt.ylabel('Volumic energy ($kg.m^{-1}.s^{-2}$)'); plt.xlabel("Radius (m)")
plt.xlim(x[0], x[len(x)-1])
plt.grid()
plt.legend()

plt.subplot(223)
plt.plot(r, P_1d, label="Solver")
plt.plot(rt, P_exact, label=f"Exact")
plt.ylabel('Pressure ($kg.m^{-1}.s^{-2}$)'); plt.xlabel("Radius (m)")
plt.xlim(x[0], x[len(x)-1])
plt.grid()
plt.legend()

plt.figure(figsize=(10,8))
plt.plot(r, P_1d, 'x', label="Solver")
plt.plot(rt, P_exact, label=f"Exact")
plt.ylabel('Pressure ($kg.m^{-1}.s^{-2}$)'); plt.xlabel("Radius (m)")
plt.xlim(1, x[len(x)-1])
plt.grid()
plt.legend()

#plt.show()

