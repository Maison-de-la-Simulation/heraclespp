# Test the advection crenel and compare 

import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys

from param import *

print("********************************")
print(" Advection test : crenel")
print("********************************")

# Solution solver 
filename = sys.argv[1]

with h5py.File(str(filename), 'r') as f : 
    print(f.keys())
    rho = f['rho'][:, 0, 0]

dx = L / len(rho)
x = np.zeros(len(rho))
for i in range(0, len(rho)):
    x[i] = i * dx + dx / 2

plt.figure(figsize=(10,8))
plt.title('Advection test')
plt.plot(x_ad, rho0_ad_cre, '--', label='t = 0')
plt.plot(x, rho, label='Solveur t=1')
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
plt.ylabel('Density'); plt.xlabel('Position')
plt.grid()
plt.legend()
plt.show()
