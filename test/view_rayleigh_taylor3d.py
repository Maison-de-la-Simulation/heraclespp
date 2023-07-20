# View Rayleigh Taylor instability

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py
import sys

print("********************************")
print("Rayleigh Taylor instability 3d")
print("********************************")

filename = sys.argv[1]

with h5py.File(str(filename), 'r') as f :
    print(f.keys())
    fx = f['fx'][0, :, 0, :]
    rho = f['rho'][: , 0, :]
    x = f['x'][()]
    y = f['y'][()]
    z = f['z'][()]
    t = f['current_time'][()]
    iter = f['iter'][()]

print("Final time =", t, "s")
print("Iteration number =", iter )

# ------------------------------------------

plt.figure(figsize=(10,8))
plt.suptitle('Rayleigh Taylor instability')
plt.title(f'Density t = {t:1f} s')
plt.imshow(rho, cmap='seismic', origin='lower', extent=[np.min(x), np.max(x), np.min(z), np.max(z)])
plt.colorbar()
plt.xlabel('x'); plt.ylabel('y')

plt.figure(figsize=(10,8))
plt.suptitle('Rayleigh Taylor instability')
plt.title(f'Passive scalar t = {t:1f} s')
plt.imshow(fx, cmap='seismic', origin='lower', extent=[np.min(x), np.max(x), np.min(z), np.max(z)])
plt.colorbar()
plt.xlabel('x'); plt.ylabel('z')
plt.show()
