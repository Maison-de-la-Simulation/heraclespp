# View stratified atmosphere

import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys

print("*************************************")
print("Spherical hydrodynamique equilibrum")
print("*************************************")

filename = sys.argv[1]

with h5py.File(str(filename), 'r') as f :
    #print(f.keys())
    rho = f['rho'][0, 0, :]
    u = f['ux'][0, 0, :]
    P = f['P'][0, 0, :]
    x = f['x'][()]
    t = f['current_time'][()]
    iter = f['iter'][()]
    gamma = f['gamma'][()]

L = np.max(x) - np.min(x)

dx = np.zeros(len(rho))
for i in range(len(rho)):
    dx[i] = x[i+1] - x[i]

xc = np.zeros(len(rho))
for i in range(len(rho)):
    xc[i] = x[i] + dx[i] / 2

print("Final time =", t, "s")
print("Iteration number =", iter)

# Analytical result ------------------------

kb = 1.38e-23
mp = 1.67e-27
G = 6.6743015e-11
mu = 1
M = 2e19
g = np.zeros(len(rho))
for i in range(len(g)):
    g[i] = G * M / x[i]**2
print("g =", g)
g2 = G * M
g0 = G * M / x[0]**2

rho0 = 10
T = 100
x0 = np.zeros(len(rho))
for i in range(len(x0)):
    x0[i] = kb * T / (mu * mp * g2)
P0 = rho0 * kb * T / (mu * mp)

rho_tab0 = rho0 * np.exp(1 / (x0 * xc))
P_tab0 = rho_tab0 * kb * T / (mp * mu)
u_tab0 = np.zeros(len(u))

print("cs = ", np.sqrt(gamma * P0 / rho0))
print("tff = ", np.sqrt(2 * L / g0))
print("h = ", x0[0])

# ------------------------------------------

#plt.figure(figsize=(15,5))
plt.figure(figsize=(10,8))
plt.suptitle('Hydrodynamic equilibrum')
plt.subplot(221)
plt.plot(xc / np.max(xc), rho_tab0, label='t = 0')
plt.plot(xc / np.max(xc), rho, 'x', label =f't = {t:1f}')
plt.xlabel('x / $x_{max}$'); plt.ylabel('Density ($kg.m^{-3}$)')
plt.yscale('log')
plt.grid()
plt.legend()
plt.subplot(222)
plt.plot(xc / np.max(xc), P_tab0,  label='t = 0')
plt.plot(xc / np.max(xc), P, label = f't = {t:1f}')
plt.xlabel('x /$x_{max}$'); plt.ylabel('Pressure ($kg.m^{-1}.s^{-2}$)')
plt.yscale('log')
plt.legend()
plt.grid()
plt.subplot(223)
plt.plot(xc / np.max(xc), u_tab0,  label='t = 0')
plt.plot(xc / np.max(xc), u, label = f't = {t:1f}')
#plt.plot(xc / np.max(xc), u / cs, label = f't = {t:1f}')
plt.xlabel('x /$x_{max}$'); plt.ylabel('Velocity ($m.s^{-1}$)')
plt.legend()
plt.grid()
plt.show()
