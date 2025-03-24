# Test the advection sinusoide and compare

import h5py
import matplotlib.pyplot as plt
import numpy as np
import sys

print("********************************")
print("  Advection test : sinus")
print("********************************")

filename = sys.argv[1]

with h5py.File(str(filename), 'r') as f:
    rho = f['rho'][0, 0, :]
    x = f['x'][()]
    t = f['current_time'][()]
    iter = f['iter'][()]

print(f"Final time = {t:.1f} s")
print(f"Iteration number = {iter}")

xmin = x[2]
xmax = x[len(x)-3]
L = xmax - xmin
dx = L / len(rho)

xc = np.zeros(len(rho))
for i in range(2, len(rho)+2):
    xc[i-2] = x[i] + dx / 2

# Analytical result ------------------------

rho0 = np.zeros(len(rho))

for i in range(len(rho0)):
    rho0[i] = 1 + 0.1 * np.sin(2 * np.pi * xc[i])

# ------------------------------------------

plt.figure(figsize=(10,8))
plt.title('Sinus advection test')
plt.plot(xc, rho0, '--', label='t = 0')
plt.plot(xc, rho, label=f't = {t:.1f}')
plt.xlabel('Position')
plt.ylabel('Density')
plt.legend(frameon=False)

plt.savefig("advection_sinus.pdf")

plt.show()
