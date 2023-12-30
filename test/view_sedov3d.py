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
    P_3d = f['P'][:, :, :] # P(phi, theta, r)
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

# Analytical result 1d ------------------------

rho0 = 1
E0 = 1

n = 3 # spherical

r_exact, rho_exact, u_exact, P_exact = ExactSedov(rho0, E0, t, gamma, n)
E_exact = 1 / 2 * rho_exact * u_exact**2 + P_exact / (gamma - 1)

rt = r_exact+1

# Analytical result 2d ------------------------

beta = 1.15 #gamma = 5/3

r_choc = beta * (E0 * t**2 / rho0)**(1/5)

theta = np.linspace(0, 2*np.pi, len(x))
x_choc = r_choc * np.sin(theta)
z_choc = r_choc * np.cos(theta)+1

# Distance from the explosion -------------------

# explosion corrdinates
x0 = 1 * np.sin(np.pi / 2) * np.cos(np.pi / 2)
y0 = 1 * np.sin(np.pi / 2) * np.sin(np.pi / 2)
z0 = 1 * np.cos(np.pi / 2)

""" plt.figure(figsize=(10,8))
plt.xlabel("Radius (m)"); plt.ylabel('Pressure ($kg.m^{-1}.s^{-2}$)')
for i in range(nr):
    print(i)
    for j in range(nth):
        k = int(nphi / 2)
        xcart = r[i] * np.sin(th[j]) * np.cos(phi[k])
        ycart = r[i] * np.sin(th[j]) * np.sin(phi[k])
        zcart = r[i] * np.cos(th[j])
        rdist = np.sqrt((xcart - x0)**2 + (ycart - y0)**2 + (zcart - z0)**2)
        plt.plot(rdist, P_3d[k, j, i], "x", color='#1f77b4')
plt.plot(r_exact, P_exact, label=f"Exact", color='#ff7f0e')
plt.legend()
plt.xlim(0, 1)
plt.show() """

# Spherical to cartesian -------------------

R, Theta = np.meshgrid(r, th)

X = R * np.sin(Theta)
Z = R * np.cos(Theta)

plt.figure(figsize=(10, 8))
plt.suptitle('Sedov blast wave spherical 3d in Cartesian coordinates')
plt.plot(x_choc, z_choc, '--', c='black')
plt.contourf(Z, X, rho_r_th, cmap='gnuplot')
plt.gca().set_aspect('equal')
plt.xlabel('x'); plt.ylabel('z')
plt.colorbar(shrink=0.55)
plt.savefig("sedov3d.pdf")
plt.show()

# ------------------------------------------

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



