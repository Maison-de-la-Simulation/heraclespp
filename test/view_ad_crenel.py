# Test the advection crenel and compare 

import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys

from param import *

print("********************************")
print(" Advection test : crenel")
print("********************************")

filename = sys.argv[1]

with h5py.File(str(filename), 'r') as f : 
    print(f.keys())
    rho = f['rho'][0, 0, :]
    x = f['x'][()]
    t = f['current_time'][()]
    iter = f['iter'][()]

dx = L / len(rho)

xc = np.zeros(len(rho))
for i in range(len(rho)):
    xc[i] = x[i] + dx / 2

print("Final time =", t, "s")
print("Iteration number =", iter )

# ------------------------------------------

plt.figure(figsize=(10,8))
plt.title('Advection test')
plt.plot(x_ad, rho0_ad_cre, '--', label='t = 0')
plt.plot(xc, rho, label='Solveur t=1')
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
plt.ylabel('Density'); plt.xlabel('Position')
plt.grid()
plt.legend()
plt.show()
