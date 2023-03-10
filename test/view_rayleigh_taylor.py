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
    u_x = f['u'][0, 0, :, :]
    u_y = f['u'][1, 0, :, :]
    P = f['P'][0, :, :]

plt.figure(figsize=(10,8))
plt.suptitle('Rayleigh Taylor instability')
#plt.subplot(121)
plt.title('Density')
plt.imshow(rho.T, origin='lower', extent=[-0.25, 0.25, -0.75, 0.75])
plt.colorbar()
plt.inferno()
plt.xlabel('x'); plt.ylabel('y')
""" plt.subplot(122)
plt.title('Pressure')
plt.imshow(P.T, origin='lower', extent=[-0.25, 0.25, -0.75, 0.75])
plt.colorbar()
plt.inferno()
plt.xlabel('x'); plt.ylabel('y') """
plt.show() 