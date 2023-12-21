# View Rayleigh Taylor instability

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py
import sys

print("********************************")
print("   Rayleigh Taylor spherical 3d")
print("********************************")

s_theta = 100
s_phi = 100

with h5py.File("test_00000000.h5", 'r') as f :
    rho0_1d = f['rho'][int(s_phi/2), int(s_theta/2), :] # rho(r)
    v0_1d = f['ux'][int(s_phi/2), int(s_theta/2), :] # vr(r)

filename = sys.argv[1]

with h5py.File(str(filename), 'r') as f :
    #print(f.keys())
    rho_1d = f['rho'][int(s_phi/2), int(s_theta/2), :] # rho(r)
    v_1d = f['ux'][int(s_phi/2), int(s_theta/2), :] # vr(r)
    rho2 = f['rho'][:, 0, :] # rho(r, phi)
    x = f['x'][()]
    y = f['y'][()]
    z = f['z'][()]
    t = f['current_time'][()]
    iter = f['iter'][()]
    gamma = f['gamma'][()]

print(f"Final time = {t:.5f} s")
print(f"Iteration number = {iter}")

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
    rc[i-2] = x[i] + dr / 2

# ------------------------------------------

plt.figure(figsize=(10,8))
plt.suptitle('Rayleigh Taylor spherical 3D')
plt.title(f'Density t = {t:.5f} s')
plt.imshow(rho2, cmap='seismic', origin='lower', extent=[rmin, rmax, phi_min, phi_max])
plt.xlabel("Radius (m)"); plt.ylabel(r"$\phi$ angle (rad)")
plt.colorbar()

plt.figure(figsize=(10,8))
plt.subplot(121)
plt.plot(rc, rho0_1d, "--", label="t = 0")
plt.plot(rc, rho_1d, label=f"t = {t:.3f}")
plt.xlabel("Radius (m)"); plt.ylabel(r"Density")
plt.xlim(rmin, rmax)
plt.legend()

plt.subplot(122)
plt.plot(rc, v0_1d, "--", label="t = 0")
plt.plot(rc, v_1d, label=f"t = {t:.3f}")
plt.xlabel("Radius (m)"); plt.ylabel(r"Velocity")
plt.xlim(rmin, rmax)
plt.xlim(x[0], x[len(x)-1])
plt.legend()

plt.show()
