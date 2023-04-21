# View stratified atmosphere

import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys

print("********************************")
print("Stratified atmosphere")
print("********************************")

filename = sys.argv[1]

with h5py.File(str(filename), 'r') as f :
    #print(f.keys())
    rho = f['rho'][0, 0, :]
    u = f['u'][0, 0, :][0]
    P = f['P'][0, 0, :]
    x = f['x'][()]
    t = f['current_time'][()]
    iter = f['iter'][()]
    gamma = f['gamma'][()]

L = np.max(x) - np.min(x)

dx = L / len(rho)

xc = np.zeros(len(rho))
for i in range(len(rho)):
    xc[i] = x[i] + dx / 2

print("Final time =", t, "s")
print("Iteration number =", iter )


# Analytical result ------------------------

rho0 = 10
kb = 1.38e-23
mh = 1.67e-27
T = 100
g = 10
mu = 1
x0 = kb * T / (mu * mh * np.abs(g))
P0 = rho0 * kb * T / (mu * mh)

rho_tab0 = rho0 * np.exp(xc / x0)
P_tab0 = rho_tab0 * kb * T / (mh * mu)
u_tab0 = np.zeros(len(u))

cs = np.sqrt(gamma * P0 / rho0)
print("cs = ", cs)
print("tff = ", np.sqrt(2*L / np.abs(g)))

# ------------------------------------------

plt.figure(figsize=(15,5))
plt.suptitle('Stratified atmopshere')
plt.subplot(131)
#plt.plot(xc / np.max(xc), rho_tab0,  label='t = 0')
plt.plot(xc / np.max(xc), np.abs(rho - rho_tab0) / rho_tab0,  label =f't = {t:1f}')
plt.xlabel('x / $x_{max}$'); plt.ylabel('Density')
plt.yscale('log')
plt.grid()
plt.legend()
plt.subplot(132)
#plt.plot(xc / np.max(xc), P_tab0,  label='t = 0')
plt.plot(xc / np.max(xc), np.abs(P - P_tab0) / P_tab0, label = f't = {t:1f}')
plt.xlabel('x /$x_{max}$'); plt.ylabel('Pressure')
plt.yscale('log')
plt.legend()
plt.grid()
plt.subplot(133)
#plt.plot(xc / np.max(xc), u_tab0,  label='t = 0')
plt.plot(xc / np.max(xc), u / cs, label = f't = {t:1f}')
#plt.plot(P / rho)
plt.xlabel('x /$x_{max}$'); plt.ylabel('Velocity')
plt.legend()
plt.grid()
plt.show()