# View Rayleigh Taylor instability

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py
import sys

print("********************************")
print("   Shock tube spherical 3d")
print("********************************")

with h5py.File("test_00000000.h5", 'r') as f :
    #print(f.keys())
    rho0 = f['rho'][0 , 0, :] # rho(r)
    u0 = f['ux'][0, 0, :]
    P0 = f['P'][0, 0, :]

with h5py.File("tube_sph.h5", 'r') as f :
    rho_tube = f['rho'][0 , 0, :] # rho(r)
    u_tube = f['ux'][0, 0, :]
    P_tube = f['P'][0, 0, :]
    xtube = f['x'][()]
    t = f['current_time'][()]
    iter = f['iter'][()]

filename = sys.argv[1]

with h5py.File(str(filename), 'r') as f :
    #print(f.keys())
    rho_1d = f['rho'][5 , 5, :] # rho(r)
    rho2 = f['rho'][:, 0, :] # rho(r, phi)
    u_1d = f['ux'][0, 0, :]
    P_1d = f['P'][0, 0, :]
    x = f['x'][()]
    y = f['y'][()]
    z = f['z'][()]
    t = f['current_time'][()]
    iter = f['iter'][()]
    gamma = f['gamma'][()]

print(f"Final time = {t:.1f} s")
print(f"Iteration number = {iter}")

rmin = x[2]
rmax = x[len(x)-3]
th_min = y[2]
th_max = y[len(y)-3]
phi_min = z[2]
phi_max = z[len(z)-3]

nr = len(rho_1d)

dr = (rmax - rmin) / nr
dth = (th_max - th_min) / nr
dphi = (phi_max - phi_min) / nr

rc = np.zeros(nr)
thc = np.zeros(nr)
phic = np.zeros(nr)
for i in range(2, nr+2):
    rc[i-2] = x[i] + dr / 2
    """ thc[i-2] = y[i] + dth / 2
    phic[i-2] = z[i] + dphi / 2 """

xmin = xtube[2]
xmax = xtube[len(xtube)-3]
L = xmax - xmin

dx = np.zeros(len(rho_tube))
for i in range(2, len(rho_tube)+2):
    dx[i-2] = xtube[i+1] - xtube[i]

xc = np.zeros(len(rho_tube))
for i in range(2, len(rho_tube)+2):
    xc[i-2] = xtube[i] + dx[i-2] / 2

e0 = P0 / rho0 / (gamma - 1)
e_tube = P_tube / rho_tube / (gamma - 1)
e_1d = P_1d / rho_1d / (gamma - 1)

# ------------------------------------------

plt.figure(figsize=(10,8))
plt.suptitle('Shock tube spherical 3D')
plt.title(f'Density t = {t:.1f} s')
plt.imshow(rho2, cmap='seismic', origin='lower', extent=[rmin, rmax, phi_min, phi_max])
plt.xlabel("Radius (m)"); plt.ylabel(r"$\phi$ angle (rad)")
plt.colorbar()

plt.figure(figsize=(10,8))
plt.suptitle(f'Shock tube t = {t:.1f} s')
plt.subplot(221)
plt.plot(rc, rho0, "--", label="t=0")
plt.plot(rc, rho_1d, label="3D sphérique")
plt.plot(xc, rho_tube, label="1D sphérique")
plt.grid()
plt.xlabel('Radius (m)');plt.ylabel(r'Density ($kg.m^{-3}$)')
plt.legend()

plt.subplot(222)
plt.plot(rc, u0, "--", label="t=0")
plt.plot(rc, u_1d, label="3D sphérique")
plt.plot(xc, u_tube, label="1D sphérique")
plt.grid()
plt.xlabel('Radius (m)');plt.ylabel(r'Velocity ($m.s^{-1}$)')
plt.legend()

plt.subplot(223)
plt.plot(rc, P0, "--", label="t=0")
plt.plot(rc, P_1d, label="3D sphérique")
plt.plot(xc, P_tube, label="1D sphérique")
plt.grid()
plt.xlabel('Radius (m)');plt.ylabel(r'Pressure ($kg.m^{-1}.s^{-2}$)')
plt.legend()

plt.subplot(224)
plt.plot(rc, e0, "--", label="t=0")
plt.plot(rc, e_1d, label="3D sphérique")
plt.plot(xc, e_tube, label="1D sphérique")
plt.grid()
plt.xlabel('Radius (m)');plt.ylabel(r'Internal energy ($m^{2}.s^{-2}$)')
plt.legend()

plt.show()
