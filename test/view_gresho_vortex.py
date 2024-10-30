# View Gresho Vortex problem

import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np

print("********************************")
print("        Gresho Vortex")
print("********************************")

filename = sys.argv[1]

with h5py.File(str(filename), 'r') as f:
    #print(f.keys())
    u_x = f['ux'][0, :, :]
    u_y = f['uy'][0, :, :]
    t = f['current_time'][()]
    iter = f['iter'][()]

print(f"Final time = {t:.1f} s")
print(f"Iteration number = {iter}")

u2d = np.sqrt(u_x**2 + u_y**2)

# ------------------------------------------

plt.figure(figsize=(15, 8))
plt.suptitle('Gresho Vortex')
plt.title(f'Speed t = {t:.1e} s')
plt.imshow(u2d, origin='lower', extent=[-1, 1, -1, 1])
plt.colorbar()
plt.plasma()
plt.xlabel('x')
plt.ylabel('y')
plt.show()
