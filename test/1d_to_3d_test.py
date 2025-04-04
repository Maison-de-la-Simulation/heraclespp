# SPDX-FileCopyrightText: 2025 The HERACLES++ development team, see COPYRIGHT.md file
#
# SPDX-License-Identifier: MIT

import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np

print("********************************")
print("         1d to 3d test")
print("********************************")

file_init = '../src/setups/1d_to_3d_test/test_1d_to_3d.h5'
with h5py.File(str(file_init), 'r') as f:
    rho0 = f['rho_1d'][()]
    u0 = f['u_1d'][()]
    P0 = f['P_1d'][()]
    x = f['x'][()]
    t0 = f['current_time'][()]

file_final = "../src/setups/v1d_500/v1d_1e5.h5"
with h5py.File(str(file_final), 'r') as f:
    rhof = f['rho'][0, 0, :]
    uf = f['ux'][0, 0, :]
    Pf = f['P'][0, 0, :]
    Tf = f['T'][0, 0, :]
    xf = f['x'][()]
    tf = f['current_time'][()]

filename = sys.argv[1]
with h5py.File(str(filename), 'r') as f:
    print(f.keys())
    rho = f['rho'][0, 0, :]
    rho2 = f['rho'][0, :, :]
    u = f['ux'][0, 0, :]
    P = f['P'][0, 0, :]
    E = f['E'][0, 0, :]
    T = f['T'][0, 0, :]
    x2 = f['x'][()]
    y = f['y'][()]
    t = f['current_time'][()]
    iter = f['iter'][()]
    gamma = f['gamma'][()]

tday = t / 3600 / 24

print(f"Final time = {t:.3e} s or {tday} days")
print(f"Iteration number = {iter:3e}")

nx = len(rho0)
dx = np.zeros(nx)
for i in range(2, nx+2):
    dx[i-2] = x[i+1] - x[i]
xc = np.zeros(nx)
for i in range(2, nx+2):
    xc[i-2] = x[i] + dx[i-2] / 2

dxf = np.zeros(len(rhof))
for i in range(2, len(rhof)+2):
    dxf[i-2] = xf[i+1] - xf[i]
xcf = np.zeros(len(rhof))
for i in range(2, len(rhof)+2):
    xcf[i-2] = xf[i] + dxf[i-2] / 2

# cgs units --------------------------------

xc_cm = xc * 10**2

rho0_cgs = rho0 * 10**(-3)
u0_cgs = u0 * 10**2
P0_cgs = P0 * 10

rho_cgs = rho * 10**(-3)
u_cgs = u * 10**2
P_cgs = P * 10
rho2_cgs = rho2 * 10**(-3)

rhof_cgs = rhof * 10**(-3)
uf_cgs = uf * 10**2
Pf_cgs = Pf * 10
xcf_cm = xcf * 10**2

# ------------------------------------------

plt.figure(figsize=(12,8))
plt.suptitle(f'Loglog graph for v1d 1e5, t = {t:.1e} s ({tday:1f} jours)')
plt.subplot(221)
plt.plot(xc_cm, rho0_cgs, "--", label=f"$t_0$= {t0:.1e} s")
plt.plot(xc_cm, rho_cgs, color='green', label=f"$t$= {t:.1e} s")
plt.plot(xcf_cm, rhof_cgs, "--", color="red", label=f"$t$= {tf:.1e} s")
plt.xlabel('rc (cm)')
plt.ylabel('Density ($g.cm^{-3}$)')
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.legend()

plt.subplot(222)
plt.plot(xc_cm, u0_cgs, "--", label=f"$t_i$= {t0:.1e} s")
plt.plot(xcf_cm, uf_cgs, "--", color="red", label=f'$t_f$ = {tf:.1e} s')
plt.plot(xc_cm, u_cgs, color='green', label=f'$t$ = {t:.1e} s')
plt.xlabel('rc (cm)')
plt.ylabel('Velocity ($cm.s^{-1}$)')
plt.xscale('log')#; plt.yscale('log')
plt.grid()
plt.legend()

plt.subplot(223)
plt.plot(xc_cm, P0_cgs, "--", label=f"$t_i$= {t0:.1e} s")
plt.plot(xcf_cm, Pf_cgs, "--", color="red", label=f'$t_f$ = {tf:.1e} s')
plt.plot(xc_cm, P_cgs, color='green', label=f'$t$ = {t:.1e} s')
plt.xlabel('rc (cm)')
plt.ylabel('Pressure ($g.cm^{-1}.s^{-2}$)')
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.legend()

plt.subplot(224)
plt.plot(xcf_cm, Tf, "--", color="red", label=f'$t_f$ = {tf:.1e} s')
plt.plot(xc_cm, T, color='green', label=f'$t$ = {t:.1e} s')
plt.xlabel('rc (cm)')
plt.ylabel('Temperature ($K$)')
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.legend()

plt.figure(figsize=(12,8))
plt.imshow(rho2_cgs, extent=[np.min(xc_cm), np.max(xc_cm), np.min(xc_cm), np.max(xc_cm)])
plt.xlabel('rc (cm)')
plt.ylabel('theta')
plt.colorbar()
plt.show()
