# Test the advection crenel and compare 

import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys

print("********************************")
print("     Advection test : gap")
print("********************************")

filename = sys.argv[1]

with h5py.File(str(filename), 'r') as f : 
    print(f.keys())
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

nx = 100
dx_an = L / nx
x_ad = np.zeros(nx)

for i in range(nx):
    x_ad[i] = i * dx_an + dx_an / 2

rho0 = np.zeros(nx)
for i in range(nx):
    if (x_ad[i] < 0.3) :
        rho0[i] = 1
    elif (x_ad[i] > 0.7):
        rho0[i] = 1
    else :
        rho0[i] = 2

# ------------------------------------------

plt.figure(figsize=(10,8))
plt.title('Gap advection test')
plt.plot(x_ad, rho0, '--', label='t = 0')
plt.plot(xc, rho, label=f't = {t:.1f}')
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
plt.ylabel('Density'); plt.xlabel('Position')
plt.grid()
plt.legend()
plt.show()
