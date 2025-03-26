# View stratified atmosphere

import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np

print("*************************************")
print("Spherical hydrodynamique equilibrium")
print("*************************************")

filename = sys.argv[1]

with h5py.File(str(filename), 'r') as f:
    rho = f['rho'][0, 0, :]
    u = f['ux'][0, 0, :]
    P = f['P'][0, 0, :]
    x = f['x_ng'][()]
    t = f['current_time'][()]
    gamma = f['gamma'][()]

L = x[-1] - x[0]
xc = (x[:-1] + x[1:]) / 2

print(f"Final time = {t:.1f} s")

# Analytical result ------------------------

kb = 1.38e-23 # m^{2}.kg.s^{-2}.K^{-1}
mp = 1.67e-27 # kg
G = 6.6743015e-11 # kg
mu = 1
M = 2e19 # kg

rho0 = 10 # kg.m^{-3}
T = 100 # K

g = G * M / xc**2
g2 = G * M
g0 = G * M / xc[0]**2

x0 = np.zeros(len(rho))
for i in range(len(x0)):
    x0[i] = kb * T / (mu * mp * g2)
P0 = rho0 * kb * T / (mu * mp) # kg.m^{-1}.s^{-2}

rho_tab0 = rho0 * np.exp(1 / (x0 * xc))
P_tab0 = rho_tab0 * kb * T / (mp * mu)
u_tab0 = np.zeros(len(u))

print(f"Sound speed = {np.sqrt(gamma * P0 / rho0):.1f} s")
print(f"Free fall time = {np.sqrt(2 * L / g0):.1f} s")
print(f"Scale = {x0[0]:.1e} m")

# ---------------------------------------------------------------------------- #

plt.figure(figsize=(12,8))
plt.suptitle('Spherical hydrodynamic equilibrium')
plt.subplot(221)
plt.plot(xc / np.max(xc), rho_tab0, "--", lw=2, c="black", label='$t=0$ s')
plt.plot(xc / np.max(xc), rho, 'x', label =f't = {t:.1f} s')
plt.xlabel('$x / x_{max}$')
plt.ylabel('Density ($kg.m^{-3}$)')
plt.yscale('log')
plt.legend(frameon=False)

plt.subplot(222)
plt.plot(xc / np.max(xc), P_tab0, "--", lw=2, c="black", label='$t=0$ s')
plt.plot(xc / np.max(xc), P, label = f't = {t:.1f} s')
plt.xlabel('x /$x_{max}$')
plt.ylabel('Pressure ($kg.m^{-1}.s^{-2}$)')
plt.yscale('log')
plt.legend(frameon=False)

plt.subplot(223)
plt.plot(xc / np.max(xc), u_tab0, "--", lw=2, c="black", label='$t=0$ s')
plt.plot(xc / np.max(xc), u, label = f't = {t:.1f} s')
plt.xlabel('$x /x_{max}$')
plt.ylabel('Velocity ($m.s^{-1}$)')
plt.legend(frameon=False)

plt.show()
