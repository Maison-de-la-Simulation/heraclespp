# View Rayleigh Taylor instability

import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys

print("********************************")
print("Rayleigh Taylor instability")
print("********************************")

filename = sys.argv[1]

with h5py.File(str(filename), 'r') as f :
    print(f.keys())
    rho = f['rho'][0, :, :]
    u = f['ux'][0, :, :]
    P = f['P'][0, :, :]
    x = f['x'][()]
    y = f['y'][()]
    fx = f['fx'][0, 0, :, :]
    t = f['current_time'][()]
    iter = f['iter'][()]

print("Final time =", t, "s")
print("Iteration number =", iter )

print("Max fx =", np.max(fx))

print(u)

# ------------------------------------------

plt.figure(figsize=(10,8))
plt.suptitle('Rayleigh Taylor instability')
plt.title(f'Density t = {t:1f} s')
plt.imshow(rho, origin='lower', extent=[np.min(x), np.max(x), np.min(y), np.max(y)])
plt.colorbar()
plt.inferno()
plt.xlabel('x'); plt.ylabel('y')

plt.figure()
plt.title('Passive scalar')
plt.imshow(fx, origin='lower', extent=[np.min(x), np.max(x), np.min(y), np.max(y)])
plt.colorbar()
plt.show() 
