# View Kelvin-Helmholtz instability

import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys

print("********************************")
print(" Kelvin-Helmholtz instability")
print("********************************")

filename = sys.argv[1]

with h5py.File(str(filename), 'r') as f :
    print(f.keys())
    rho = f['rho'][0, :, :]
    x = f['x'][()]
    y = f['y'][()]
    t = f['current_time'][()]
    iter = f['iter'][()]

print("Final time =", t, "s")
print("Iteration number =", iter )

# ------------------------------------------

plt.figure(figsize=(10,8))
plt.suptitle('Kelvin-Helmholtz instability')
plt.title(f'Density t = {t:1f} s')
plt.imshow(rho, cmap='seismic', origin='lower', extent=[np.min(x), np.max(x), np.min(y), np.max(y)])
plt.colorbar(shrink=0.5)
plt.xlabel('x'); plt.ylabel('y')
plt.show() 