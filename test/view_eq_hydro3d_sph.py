# View hydrostaic equilibrum pherical 3d

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py
import sys

print("*************************************")
print("Spherical 3d hydrodynamique equilibrum")
print("*************************************")

with h5py.File("test_00000000.h5", 'r') as f :
    rho0 = f['rho'][5, 5, :]
    u0 = f['ux'][5, 5, :]
    P0 = f['P'][5, 5, :]

filename = sys.argv[1]

with h5py.File(str(filename), 'r') as f :
    rho = f['rho'][5, 5, :]
    u = f['ux'][5, 5, :]
    P = f['P'][5, 5, :]
    x = f['x'][()]
    t = f['current_time'][()]
    iter = f['iter'][()]
    gamma = f['gamma'][()]

print(f"Final time = {t:.5f} s")
print(f"Iteration number = {iter}")

nr = len(rho)
rmin = x[2]
rmax = x[len(x)-3]

dr = (rmax - rmin) / nr
rc = np.zeros(nr)
for i in range(2, nr+2):
    rc[i-2] = x[i] + dr / 2

# ------------------------------------------

plt.figure(figsize=(10,8))
plt.suptitle('Hydrodynamic equilibrum 3d spherical')
plt.subplot(221)
plt.plot(rc / np.max(rc), rho0, "--", label='t = 0 s')
plt.plot(rc / np.max(rc), rho, 'x', label =f't = {t:.5f} s')
plt.xlabel('x / $x_{max}$'); plt.ylabel('Density ($kg.m^{-3}$)')
plt.yscale('log')
plt.grid()
plt.legend()
plt.subplot(222)
plt.plot(rc / np.max(rc), P0, "--", label='t = 0 s')
plt.plot(rc / np.max(rc), P, label = f't = {t:.5f} s')
plt.xlabel('x /$x_{max}$'); plt.ylabel('Pressure ($kg.m^{-1}.s^{-2}$)')
plt.yscale('log')
plt.legend()
plt.grid()
plt.subplot(223)
plt.plot(rc / np.max(rc), u0, "--", label='t = 0 s')
plt.plot(rc / np.max(rc), u, label = f't = {t:.5f} s')
#plt.plot(rc / np.max(rc), u / cs, label = f't = {t:1f}')
plt.xlabel('x /$x_{max}$'); plt.ylabel('Velocity ($m.s^{-1}$)')
plt.legend()
plt.grid()
plt.show()