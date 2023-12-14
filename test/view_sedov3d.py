# View Rayleigh Taylor instability

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py
import sys

print("********************************")
print("   Sedov blast wave 3d")
print("********************************")

filename = sys.argv[1]

ind_th = 10
ind_ph = 50

with h5py.File(str(filename), 'r') as f :
    #print(f.keys())
    rho_1d = f['rho'][ind_ph, ind_th, :] # rho(r)
    u_1d = f['ux'][ind_ph, ind_th, :] # u(r)
    P_1d = f['P'][ind_ph, ind_th, :] # P(r)
    rho_r_th = f['rho'][ind_ph, :, :] # rho(r, theta)
    rho_r_ph = f['rho'][:, ind_th, :] # rho(r, theta)
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
rc = np.zeros(nr)
for i in range(2, nr+2):
    rc[i-2] = x[i] #+ dr / 2

# ------------------------------------------

plt.figure(figsize=(10,8))
plt.suptitle('Rayleigh Taylor spherical 3D')
plt.title(f'Density t = {t:.5f} s')
plt.imshow(rho_r_th, cmap='seismic', origin='lower', extent=[rmin, rmax, th_min, th_max])
plt.xlabel("Radius (m)"); plt.ylabel(r"$\theta$ angle (rad)")
plt.colorbar()

plt.figure(figsize=(8,8))
plt.subplot(221)
plt.plot(rc, rho_1d, label=f"t = {t:.3f}")
plt.xlabel("Radius (m)"); plt.ylabel(r"Density")
plt.xlim(x[0], x[len(x)-1])
plt.grid()
plt.legend()

plt.subplot(222)
plt.plot(rc, E, label=f"t = {t:.3f}")
plt.ylabel('Volumic energy ($kg.m^{-1}.s^{-2}$)'); plt.xlabel("Radius (m)")
plt.xlim(x[0], x[len(x)-1])
plt.grid()
plt.legend()

plt.subplot(223)
plt.plot(rc, P_1d, label=f"t = {t:.3f}")
plt.ylabel('Pressure ($kg.m^{-1}.s^{-2}$)'); plt.xlabel("Radius (m)")
plt.xlim(x[0], x[len(x)-1])
plt.grid()
plt.legend()

plt.show()
