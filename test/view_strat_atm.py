# View stratified atmosphere

import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np

print("********************************")
print("    Stratified atmosphere")
print("********************************")

filename = sys.argv[1]

with h5py.File(str(filename), 'r') as f:
    rho = f['rho'][0, 0, :]
    u = f['ux'][0, 0, :]
    P = f['P'][0, 0, :]
    T = f['T'][0, 0, :]
    x = f['x_ng'][()]
    t = f['current_time'][()]
    iter = f['iter'][()]
    gamma = f['gamma'][()]

print(f"Final time = {t:.1f} s")
print(f"Iteration number = {iter}")

L = x[-1] - x[0]

xc = (x[:-1] + x[1:]) / 2

# Analytical result ---------------------------------------------------------- #

rho0 = 10
kb = 1.38e-23
mp = 1.67e-27
T0 = 100
g = -10
mu = 1
x0 = kb * T0 / (mu * mp * np.abs(g))
P0 = rho0 * kb * T0 / (mu * mp)

rho_tab0 = rho0 * np.exp(- xc / x0)
P_tab0 = rho_tab0 * kb * T0 / (mp * mu)
u_tab0 = np.zeros(len(u))
T_tab0 = np.ones(len(xc)) * T0

cs = np.sqrt(gamma * P0 / rho0)

print(f"Sound speed = {cs:.1f} s")
print(f"Free fall time = {np.sqrt(2*L / np.abs(g)):.1f} s")

# ------------------------------------------

plt.figure(figsize=(12,8))
plt.suptitle('Stratified atmosphere')
plt.subplot(221)
plt.plot(xc / np.max(xc), rho_tab0, "--", lw=2, c="black", label='$t=0$ s')
plt.plot(xc / np.max(xc), rho, label=f'$t = {t:.0f}$ s')
plt.xlabel('x / $x_{max}$')
plt.ylabel('Density (kg.m$^{-3}$)')
plt.yscale('log')
plt.legend(frameon=False)

plt.subplot(222)
plt.plot(xc / np.max(xc), P_tab0, "--", lw=2, c="black", label='$t=0$ s')
plt.plot(xc / np.max(xc), P, label=f'$t = {t:.0f}$ s')
plt.xlabel('x /$x_{max}$')
plt.ylabel('Pressure (kg.m$^{-1}$.s$^{-2}$)')
plt.yscale('log')
plt.legend(frameon=False)

plt.subplot(223)
plt.plot(xc / np.max(xc), u_tab0, "--", lw=2, c="black", label='$t=0$ s')
plt.plot(xc / np.max(xc), u, label=f'$t = {t:.0f}$ s')
plt.xlabel('x /$x_{max}$')
plt.ylabel('Velocity (m.s$^{-1}$)')
plt.legend(frameon=False)

plt.subplot(224)
plt.plot(xc / np.max(xc), T_tab0, "--", lw=2, c="black", label='$t=0$ s')
plt.plot(xc / np.max(xc), T, label=f'$t = {t:.0f}$ s')
plt.xlabel('x /$x_{max}$')
plt.ylabel('Temperature (K)')
plt.ylim(99.5, 100.5)
plt.legend(frameon=False)

plt.show()
