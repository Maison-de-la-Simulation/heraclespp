# Heat nickel

import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys

print("********************************")
print("         Heat nickel 56")
print("********************************")

filename = sys.argv[1]

with h5py.File(str(filename), 'r') as f : 
    #print(f.keys())
    rho = f['rho'][0, 0, :]
    u = f['ux'][0, 0, :]
    P = f['P'][0, 0, :]
    T = f['T'][0, 0, :]
    fx = f['fx'][0, 0, 0, :]
    x = f['x'][()]
    t = f['current_time'][()]
    iter = f['iter'][()]
    gamma = f['gamma'][()]

tday = t / 3600 / 24

print(f"Final time = {t:.3e} s or {tday} days")
print(f"Iteration number = {iter:3e}")
print("---------------------------")

dx = np.zeros(len(rho))
for i in range(len(rho)):
    dx[i] = x[i+1] - x[i]

xc = np.zeros(len(rho))
for i in range(len(rho)):
    xc[i] = x[i] + dx[i] / 2

# Initialisation ---------------------------

kb = 1.380649e-23 # kg.m^{2}.s^{-2}.K^{-1}
mu = 55.9421278 # 
mp = 1.672649e-27 # kg
atomic_unit_mass = 1.66053906660e-27 # kg

rho_ini = 1e-6 # kg.m^{-3} (1e-9 g.cm^{-3})
u_ini = 0 # m.s^{-1}
T_ini = 1e5 # K
fx_ini = 1 # 100%

rho0 = np.zeros(len(rho))
u0 = np.zeros(len(rho))
P0 = np.zeros(len(rho))
fx0 = np.zeros(len(rho))
T0 = np.zeros(len(rho))

for i in range(len(rho)):
    rho0[i] = rho_ini
    u0[i] = u_ini
    P0[i] = rho_ini * kb * T_ini / (mp * mu)
    fx0[i] = fx_ini
    T0[i] = T_ini

# Nickel 56 heating ------------------------

m_ni = 55.9421278 * atomic_unit_mass # kg
tau_ni = 8.8 * 3600 * 24 # s
tau_co = 111.3 * 3600 * 24 # s

mevtoJ = 1.602176634e-19 * 1e6 # J

Q_NitoCo = 1.75 * mevtoJ # J
Q_CotoFe = 3.73 * mevtoJ # J

# At t = 0
n_ni_0 = rho_ini / m_ni # m^{-3}

print(f"N Ni at t0 = {n_ni_0:.3e}")

# At t
n_ni = n_ni_0 * np.exp(-t / tau_ni) # m^{-3}
n_co = n_ni_0 * tau_co / (tau_co - tau_ni) * (np.exp(-t / tau_co) - np.exp(-t / tau_ni))
n_fe = n_ni_0 * (1 + tau_ni / (tau_co - tau_ni) * np.exp(-t / tau_ni) - tau_co / (tau_co - tau_ni) * np.exp(-t / tau_co))

print(f"N Ni = {n_ni:.3e}, N Co = {n_co:.3e}, N Fe = {n_fe:.3e}")

# Energy release
E_tot_lib = (Q_NitoCo + Q_CotoFe) * n_ni_0 # J.m^{-3}
E_ini = rho_ini * kb * T_ini / ((gamma - 1) * mu * atomic_unit_mass) # J.m^{-3}

E_final = E_ini + E_tot_lib # J.m^{-3}

T_final = (gamma - 1) * mu * atomic_unit_mass * E_final / (kb * rho_ini) # K

print(f"Release energy = {E_tot_lib:.3e}")
print(f"Total energy after release = {E_final:.3e}")
print(f"Final temperature = {T_final:.3e}")

T_ni56 = np.zeros(len(rho))
fx_ni56 = np.zeros(len(rho))

for i in range(len(rho)):
    T_ni56[i] = T_final
    fx_ni56[i] = n_ni / n_ni_0

# ------------------------------------------

plt.figure(figsize=(10,8))
plt.suptitle(f'Heat nickel t = {t:.1e} s ({tday} jours)')
plt.subplot(221)
plt.plot(xc, rho0, '--', label='t = 0')
plt.plot(xc, rho,  label = f't = {t:.1e}')
plt.ylabel('Density ($kg.m^{-3}$)'); plt.xlabel('x')
plt.yscale('log')
plt.grid()
plt.legend()

plt.subplot(222)
plt.plot(xc, u0, '--', label='t = 0')
plt.plot(xc, u,  label = f't = {t:.1e}')
plt.ylabel('Velocity ($m.s^{-1}$)'); plt.xlabel('x')
plt.grid()
plt.legend()

plt.subplot(223)
plt.plot(xc, P0, '--', label='t = 0')
plt.plot(xc, P, label = f't = {t:.1e}')
plt.ylabel('Pressure ($kg.m^{-1}.s^{-2}$)'); plt.xlabel('x')
plt.yscale('log')
plt.grid()
plt.legend()

plt.subplot(224)
plt.plot(xc, fx0, '--', label='t = 0')
plt.plot(xc, fx, label = f't = {t:.1e}')
plt.plot(xc, fx_ni56, label = 'analytical Ni56 heat')
plt.ylabel('Passiv scalar'); plt.xlabel('x')
plt.yscale('log')
plt.grid()
plt.legend()

plt.figure(figsize=(10,8))
plt.suptitle(f'Temperature for test heat nickel t = {t:.1e} s ({tday} jours)')
plt.plot(xc, T0, '--', label='t = 0')
plt.plot(xc, T, label = f't = {t:.1e}')
plt.plot(xc, T_ni56, '--', label = 'analytical Ni56 heat')
plt.ylabel('Temperature ($K$)'); plt.xlabel('x')
plt.yscale('log')
plt.grid()
plt.legend()

plt.show()
