# Test the advection sinusoide and compare 

import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys

print("********************************")
print("  Advection test : gaussian")
print("********************************")

filename = sys.argv[1]

with h5py.File(str(filename), 'r') as f : 
    print(f.keys())
    rho = f['rho'][0, 0, :]
    x = f['x'][()]
    t = f['current_time'][()]
    iter = f['iter'][()]

print("Final time =", t, "s")
print("Iteration number =", iter )

L = np.max(x) - np.min(x)

dx = L / len(rho)

xc = np.zeros(len(rho))
for i in range(len(rho)):
    xc[i] = x[i] + dx / 2
 
# Analytical result ------------------------

rho0 = np.zeros(len(rho))

for i in range(len(rho0)):
    rho0[i] = 1 * np.exp(-15 * (1. / 2  - xc[i])**2) # Sinusoïdale

# ------------------------------------------

plt.figure(figsize=(10,8))
plt.title('Advection test')
plt.plot(xc, rho0, '--', label='t = 0')
plt.plot(xc, rho, label=f't = {t:1f}')
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
plt.ylabel('Density'); plt.xlabel('Position')
plt.grid()
plt.legend()
plt.show()
