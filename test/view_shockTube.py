# Test the shcok tube problem and compare 

import numpy as np
import matplotlib.pyplot as plt
import h5py

from param import *
from shockTube_exact import CI, ExactShockTube

# Condition for pressure > 0
if (g4 * (c0l + c0r)) < (u0r - u0l) :
    print('Il faut inclure le vide')

print("********************************")
print(" Shock tube problem")
print("********************************")

# Initialisation
tab_rho0, tab_u0, tab_P0 = CI(x_exact, inter, var_int0l, var_int0r)
tab_e0 = tab_P0 / tab_rho0 / g8

# Solution Exact
timeout = 0.2
rho_exact, u_exact, P_exact, e_exact = ExactShockTube(x_exact, inter, var_int0l, var_int0r, timeout)
print('Final time shock tube problem = ', timeout, 's')

# Solution solver 
file_name = 'test_119.h5'

with h5py.File(file_name, 'r') as f : 
    print(f.keys())
    rho = f['rho'][()]
    u = f['u'][()]
    P = f['P'][()]
e = P / rho / (gamma - 1)
x = np.linspace(0, 1, len(rho))

rho_rec = rho[1:len(rho)]
u_rec = u[1:len(rho)]
P_rec = P[1:len(rho)]
e_rec = e[1:len(rho)]
x_rec = x[0:len(rho)-1]

plt.figure(figsize=(10,8))
plt.suptitle('Shock tube')
plt.subplot(221)
plt.plot(x_exact, tab_rho0, '--', label='t=0')
plt.plot(x_exact,rho_exact, label='Exact')
plt.plot(x_rec, rho_rec, label='Solver')
plt.ylabel('Densité'); plt.xlabel('Position')
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
plt.grid()
plt.legend()
plt.subplot(222)
plt.plot(x_exact, tab_u0, '--', label='t=0')
plt.plot(x_exact, u_exact, label='Exact')
plt.plot(x_rec, u_rec, label='Solver')
plt.ylabel('Speed'); plt.xlabel('Position')
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
plt.grid()
plt.legend()
plt.subplot(223)
plt.plot(x_exact, tab_P0,'--', label='t=0')
plt.plot(x_exact, P_exact, label='Exact')
plt.plot(x_rec, P_rec, label='Solver')
plt.ylabel('pressure'); plt.xlabel('Position')
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
plt.grid()
plt.legend()
plt.subplot(224)
plt.plot(x_exact, tab_e0,'--', label='t=0')
plt.plot(x_exact, e_exact, label='Exact')
plt.plot(x_rec, e_rec, label='Solver')
plt.ylabel('Internal energy'); plt.xlabel('Position')
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
plt.grid()
plt.legend()
plt.show()