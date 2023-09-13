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
    #print(f.keys())
    rho = f['rho'][0, :, :]
    x = f['x'][()]
    y = f['y'][()]
    t = f['current_time'][()]
    iter = f['iter'][()]

print(f"Final time = {t:.1f} s")
print(f"Iteration number = {iter}")

xmin = x[2]
xmax = x[len(x)-3]
ymin = y[2]
ymax = y[len(y)-3]

# ------------------------------------------

plt.figure(figsize=(10,8))
plt.suptitle('Kelvin-Helmholtz instability')
plt.title(f'Density t = {t:.1f} s')
plt.imshow(rho, cmap='seismic', origin='lower', extent=[xmin, xmax, ymin, ymax])
plt.colorbar(shrink=0.5)
plt.xlabel('x'); plt.ylabel('y')
plt.show() 