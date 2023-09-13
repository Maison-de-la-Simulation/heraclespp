# View Rayleigh Taylor instability

import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys

print("********************************")
print("Rayleigh Taylor instability 2d")
print("********************************")

filename = sys.argv[1]

with h5py.File(str(filename), 'r') as f :
    #print(f.keys())
    rho = f['rho'][0, :, :]
    u = f['ux'][0, :, :]
    P = f['P'][0, :, :]
    Py = f['P'][0, :, 0]
    x = f['x'][()]
    y = f['y'][()]
    fx = f['fx'][0, 0, :, :]
    t = f['current_time'][()]
    iter = f['iter'][()]

print(f"Final time = {t:.1f} s")
print(f"Iteration number = {iter}")

print("Max fx =", np.max(fx))

xmin = x[2]
xmax = x[len(x)-3]
ymin = y[2]
ymax = y[len(y)-3]

# ------------------------------------------

plt.figure(figsize=(10,8))
plt.suptitle('Rayleigh Taylor instability')
plt.title(f'Density t = {t:.1f} s')
plt.imshow(rho, cmap='seismic', origin='lower', extent=[xmin, xmax, ymin, ymax])
plt.colorbar()
plt.xlabel('x'); plt.ylabel('y')

plt.figure(figsize=(10,8))
plt.title('Passive scalar')
plt.imshow(fx, cmap='seismic', origin='lower', extent=[xmin, xmax, ymin, ymax])
plt.colorbar()
plt.xlabel('x'); plt.ylabel('y')
plt.show() 
