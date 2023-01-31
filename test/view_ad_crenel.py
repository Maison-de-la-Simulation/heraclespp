# Test the advection crenel and compare 

import numpy as np
import matplotlib.pyplot as plt
import h5py

from param import *

print("********************************")
print(" Advection test : crenel")
print("********************************")

# Solution solver 
file_name = 'test_358.h5'

with h5py.File(file_name, 'r') as f : 
    print(f.keys())
    rho = f['rho'][()]
x = np.linspace(0, 1, len(rho))

rho_rec = rho[2:len(rho)]
x_rec = x[0:len(rho)-2]

plt.figure(figsize=(10,8))
plt.title('Advection test')
plt.plot(x_ad[0:nx+1], rho0_ad_cre[0:nx+1], '--', label='t = 0')
plt.plot(x_rec, rho_rec, label='Solveur t=1')
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
plt.ylabel('Density'); plt.xlabel('Position')
plt.grid()
plt.legend()
plt.show()