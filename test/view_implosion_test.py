# View Liska-Wendroff implosion test

import sys
import h5py
import matplotlib.pyplot as plt

print("********************************")
print(" Liska-Wendroff implosion test")
print("********************************")

filename = sys.argv[1]

with h5py.File(str(filename), 'r') as f:
    rho = f['rho'][0, :, :]
    x = f['x_ng'][()]
    y = f['y_ng'][()]
    t = f['current_time'][()]
    iter = f['iter'][()]

print(f"Final time = {t:.1f} s")
print(f"Iteration number = {iter}")

# ------------------------------------------

plt.figure(figsize=(10,8))
plt.suptitle('View Liska-Wendroff implosion test')
plt.title(f'Density t = {t:.1f} s')
plt.imshow(rho, cmap='seismic', origin='lower', extent=[x[0], x[-1], y[0], y[-1]])
plt.colorbar()
#plt.colorbar(shrink=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
