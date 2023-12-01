# V1d 1e5

import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys

print("********************************")
print("         v1d 500")
print("********************************")

file_init = "../src/setups/v1d_500/v1d_500_logr_grid_n1000.h5"
with h5py.File(str(file_init), 'r') as f :
    rho0 = f['rho'][0, 0, :]
    u0 = f['ux'][0, 0, :]
    P0 = f['P'][0, 0, :]
    T0 = f['T'][0, 0, :]
    Ni0 = f['fx'][0, 0, 0, :]
    H0 = f['fx'][1, 0, 0, :]
    He0 = f['fx'][2, 0, 0, :]
    O0 = f['fx'][3, 0, 0, :]
    Si0 = f['fx'][4, 0, 0, :]
    t0 = f['current_time'][()]

file_final = "../test/v1d_1e5.h5"
with h5py.File(str(file_final), 'r') as f :
    rhof = f['rho'][0, 0, :]
    uf = f['ux'][0, 0, :]
    Pf = f['P'][0, 0, :]
    Tf = f['T'][0, 0, :]
    Nif = f['fx'][0, 0, 0, :]
    Hf = f['fx'][1, 0, 0, :]
    Hef = f['fx'][2, 0, 0, :]
    Of = f['fx'][3, 0, 0, :]
    Sif = f['fx'][4, 0, 0, :]
    xf = f['x'][()]
    tf = f['current_time'][()]

filename = sys.argv[1]
with h5py.File(str(filename), 'r') as f :
    print(f.keys())
    rho = f['rho'][0, 0, :]
    u = f['ux'][0, 0, :]
    P = f['P'][0, 0, :]
    E = f['E'][0, 0, :]
    T = f['T'][0, 0, :]
    Ni = f['fx'][0, 0, 0, :]
    H = f['fx'][1, 0, 0, :]
    He = f['fx'][2, 0, 0, :]
    O = f['fx'][3, 0, 0, :]
    Si = f['fx'][4, 0, 0, :]
    x = f['x'][()]
    t = f['current_time'][()]
    iter = f['iter'][()]
    gamma = f['gamma'][()]

tday = t / 3600 / 24

print(f"Final time = {t:.3e} s or {tday} days")
print(f"Iteration number = {iter:3e}")

dx = np.zeros(len(rho))
for i in range(2, len(rho)+2):
    dx[i-2] = x[i+1] - x[i]

xc = np.zeros(len(rho))
for i in range(2, len(rho)+2):
    xc[i-2] = x[i] + dx[i-2] / 2

dxf = np.zeros(len(rhof))
for i in range(2, len(rhof)+2):
    dxf[i-2] = xf[i+1] - xf[i]

xcf = np.zeros(len(rhof))
for i in range(2, len(rhof)+2):
    xcf[i-2] = xf[i] + dxf[i-2] / 2

# cgs units --------------------------------

xc_cm = xc * 10**2
xcf_cm = xcf * 10**2

rho0_cgs = rho0 * 10**(-3)
rho_cgs = rho * 10**(-3)
rhof_cgs = rhof * 10**(-3)

u0_cgs = u0 * 10**2
u_cgs = u * 10**2
uf_cgs = uf * 10**2

P0_cgs = P0 * 10
P_cgs = P * 10
Pf_cgs = Pf * 10

# Energy -------------------

ec = 1 / 2 * rho * u**2
ei = E - ec

# Mach number -------------------

kb = 1.38e-23 # kg.m^{2}.s^{-2}.K^{-1}
mh = 1.67e-27 # kg
hplanck = 6.62607015e-34 # kg.m^{2}.s^{-3}
pi = np.pi
c = 2.99792458e8 # m.s^{-1}
ar = (8 * pi**5 * kb**4) / (15 * hplanck**3 * c**3) # kg.m^{-1}.s^{-2}.K^{-4}

cs = np.zeros(len(rho))

for i in range(len(rho)):
    Pr = ar * T[i]**4 / 3
    Pg = P[i] - Pr
    alpha = Pr / Pg
    num = gamma / (gamma - 1) + 20 * alpha + 16 * alpha * alpha
    den = 1. / (gamma - 1) + 12 * alpha
    gamma_eff = num / den
    cs[i] = np.sqrt(gamma_eff * P[i] / rho[i])

""" plt.figure(figsize=(12,8))
plt.plot(xc_cm, u / cs)
plt.xlabel('rc (cm)'); plt.ylabel('Mach')
plt.xscale('log')#; plt.yscale('log') """

# ------------------------------------------

plt.figure(figsize=(12,8))
plt.suptitle(f'Loglog graph for v1d 1e5, t = {t:.1e} s ({tday:1f} jours)')
plt.subplot(221)
plt.plot(xc_cm, rho0_cgs, "--", label=f"$t_0$= {t0:.1e} s")
plt.plot(xcf_cm, rhof_cgs, "--", color="red", label=f'$t_f$ = {tf:.1e} s')
plt.plot(xc_cm, rho_cgs, color='green', label=f'$t$ = {t:.1e} s')
plt.xlabel('rc (cm)'); plt.ylabel('Density ($g.cm^{-3}$)')
plt.xscale('log'); plt.yscale('log')
plt.grid()
plt.legend()

plt.subplot(222)
plt.plot(xc_cm, u0_cgs, "--", label=f"$t_i$= {t0:.1e} s")
plt.plot(xcf_cm, uf_cgs, "--", color="red", label=f'$t_f$ = {tf:.1e} s')
plt.plot(xc_cm, u_cgs, color='green', label=f'$t$ = {t:.1e} s')
plt.xlabel('rc (cm)'); plt.ylabel('Velocity ($cm.s^{-1}$)')
plt.xscale('log')#; plt.yscale('log')
plt.grid()
plt.legend()

plt.subplot(223)
plt.plot(xc_cm, P0_cgs, "--", label=f"$t_i$= {t0:.1e} s")
plt.plot(xcf_cm, Pf_cgs, "--", color="red", label=f'$t_f$ = {tf:.1e} s')
plt.plot(xc_cm, P_cgs, color='green', label=f'$t$ = {t:.1e} s')
plt.xlabel('rc (cm)'); plt.ylabel('Pressure ($g.cm^{-1}.s^{-2}$)')
plt.xscale('log'); plt.yscale('log')
plt.grid()
plt.legend()

plt.subplot(224)
plt.plot(xc_cm, T0, "--", label=f"$t_i$= {t0:.1e} s")
plt.plot(xcf_cm, Tf, "--", color="red", label=f'$t_f$ = {tf:.1e} s')
plt.plot(xc_cm, T, color='green', label=f'$t$ = {t:.1e} s')
plt.xlabel('rc (cm)'); plt.ylabel('Temperature ($K$)')
plt.xscale('log'); plt.yscale('log')
plt.grid()
plt.legend()

plt.figure(figsize=(10,8))
plt.title(f'Loglog passiv scalar graph for v1d 1e5, t = {t:.1e} s ({tday:1f} jours)')
plt.plot(xc_cm, Ni, c='brown', label="Ni56")
plt.plot(xcf_cm, Nif, "--", c='brown', label="Ni56 V1D")
plt.plot(xc_cm, H, c="deepskyblue",label="H")
plt.plot(xcf_cm, Hf,"--", c="deepskyblue",label="H V1D")
plt.plot(xc_cm, He, c="seagreen", label="He")
plt.plot(xcf_cm, Hef, "--", c="seagreen", label="He V1D")
plt.plot(xc_cm, O, c='darkorange', label="O")
plt.plot(xcf_cm, Of, "--", color='darkorange', label="O V1D")
plt.plot(xc_cm, Si, color="violet",label="Si")
plt.plot(xcf_cm, Sif, "--", color="violet",label="Si V1D")
plt.xscale('log'); plt.yscale('log')
plt.xlabel('rc (cm)'); plt.ylabel('Concentration')
plt.grid()
plt.legend()
plt.ylim(10**(-27), 10)

"""plt.figure(figsize=(10,8))
plt.plot(xc_cm, ei / ec)
plt.xlabel('rc (cm)'); plt.ylabel('ei / ec')
plt.xscale('log'); plt.yscale('log')
plt.grid()"""

plt.show()