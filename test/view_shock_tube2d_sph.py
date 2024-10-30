import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np

print("********************************")
print("   Shock tube spherical 2d")
print("********************************")

with h5py.File("result_00000000.h5", 'r') as f:
    #print(f.keys())
    rho0 = f['rho'][0 , 0, :] # rho(r)
    u0 = f['ux'][0, 0, :]
    P0 = f['P'][0, 0, :]

filename = sys.argv[1]
ind_th = int(len(rho0) / 2)
with h5py.File(str(filename), 'r') as f:
    #print(f.keys())
    rho_2d = f['rho'][0, ind_th, :] # rho(r)
    u_2d = f['ux'][0, ind_th, :]
    P_2d = f['P'][0, ind_th, :]
    rho_2d_r_th = f['rho'][0, :, :] # rho(r, theta)
    x = f['x'][()]
    y = f['y'][()]
    t = f['current_time'][()]
    iter = f['iter'][()]
    gamma = f['gamma'][()]

print(f"Final time = {t:.1f} s")
print(f"Iteration number = {iter}")

e0 = P0 / rho0 / (gamma - 1)
e_2d = P_2d / rho_2d / (gamma - 1)

rmin = x[2]
rmax = x[len(x)-3]
th_min = y[2]
th_max = y[len(y)-3]
L = rmax - rmin
nr = len(rho_2d)

dr = (rmax - rmin) / nr
rc = np.zeros(nr)
for i in range(2, nr+2):
    rc[i-2] = x[i] + dr / 2

# ------------------------------------------

plt.figure(figsize=(10,8))
plt.suptitle('Shock tube spherical 2D')
plt.title(f'Density t = {t:.1f} s')
plt.imshow(rho_2d_r_th, cmap='seismic', origin='lower', extent=[rmin, rmax, th_min, th_max])
plt.xlabel("Radius (m)")
plt.ylabel(r"$\phi$ angle (rad)")
plt.colorbar()

plt.figure(figsize=(10,8))
plt.suptitle(f'Shock tube t = {t:.1f} s')
plt.subplot(221)
plt.plot(rc, rho0, "--", label="t=0")
plt.plot(rc, rho_2d, label="2D sphérique")
plt.grid()
plt.xlabel('Radius (m)')
plt.ylabel(r'Density ($kg.m^{-3}$)')
plt.legend()

plt.subplot(222)
plt.plot(rc, u0, "--", label="t=0")
plt.plot(rc, u_2d, label="2D sphérique")
plt.grid()
plt.xlabel('Radius (m)')
plt.ylabel(r'Velocity ($m.s^{-1}$)')
plt.legend()

plt.subplot(223)
plt.plot(rc, P0, "--", label="t=0")
plt.plot(rc, P_2d, label="2D sphérique")
plt.grid()
plt.xlabel('Radius (m)')
plt.ylabel(r'Pressure ($kg.m^{-1}.s^{-2}$)')
plt.legend()

plt.subplot(224)
plt.plot(rc, e0, "--", label="t=0")
plt.plot(rc, e_2d, label="2D sphérique")
plt.grid()
plt.xlabel('Radius (m)')
plt.ylabel(r'Internal energy ($m^{2}.s^{-2}$)')
plt.legend()

plt.show()
