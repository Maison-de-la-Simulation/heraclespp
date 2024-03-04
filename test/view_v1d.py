# V1d 1e5

import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys

print("********************************")
print("             V1D")
print("********************************")

filename = sys.argv[1]
ndim = input("Dimension of the simulation (1 or 3): ")

# ------------------------------------------

def read_start_file(filename):
    with h5py.File(str(filename), 'r') as f :
        rho = f['rho_1d'][()]
        u = f['u_1d'][()]
        P = f['P_1d'][()]
        x = f['x'][()]
        t = f['current_time'][()]
    return rho, u, P, x, t

def read_file_1d_r(filename):
    with h5py.File(str(filename), 'r') as f :
        rho = f['rho'][0, 0, :]
        u = f['ux'][0, 0, :]
        P = f['P'][0, 0, :]
        T = f['T'][0, 0, :]
        x = f['x'][()]
        t = f['current_time'][()]
    return rho, u, P, T, x, t

def read_file_1d_r_element(filename):
    with h5py.File(str(filename), 'r') as f :
        Ni = f['fx'][0, 0, 0, :]
        H = f['fx'][1, 0, 0, :]
        He = f['fx'][2, 0, 0, :]
        O = f['fx'][3, 0, 0, :]
        Si = f['fx'][4, 0, 0, :]
    return Ni, H, He, O, Si

def read_3d_file_in_1d_angular_mean(filename):
    with h5py.File(str(filename), 'r') as f :
        rho_file = f['rho'][:, :, :]
        u_file = f['ux'][:, :, :]
        P_file = f['P'][:, :, :]
        T_file = f['E'][:, :, :]
    nr = int(rho_file.shape[2])
    nth = int(rho_file.shape[1])
    nph = int(rho_file.shape[0])
    rho = np.zeros(nr)
    u = np.zeros(nr)
    P = np.zeros(nr)
    T = np.zeros(nr)
    for i in range(len(rho)):
        sum_rho = 0
        sum_u = 0
        sum_P = 0
        sum_T = 0
        for j in range(nth):
            for k in range(nph):
                sum_rho += rho_file[k, j, i]
                sum_u += u_file[k, j, i]
                sum_P += P_file[k, j, i]
                sum_T += T_file[k, j, i]
        rho[i] = sum_rho / (nth * nph)
        u[i] = sum_u / (nth * nph)
        P[i] = sum_P / (nth * nph)
        T[i] = sum_T / (nth * nph)
    return rho, u, P, T

def read_3d_file_element_in_1d_angular_mean(filename):
    with h5py.File(str(filename), 'r') as f :
        Ni_file = f['fx'][0, 0, 0, :]
        H_file = f['fx'][1, 0, 0, :]
        He_file = f['fx'][2, 0, 0, :]
        O_file = f['fx'][3, 0, 0, :]
        Si_file = f['fx'][4, 0, 0, :]
    nr = int(Ni_file.shape[2])
    nth = int(Ni_file.shape[1])
    nph = int(Ni_file.shape[0])
    rho = np.zeros(nr)
    Ni = np.zeros(nr)
    H = np.zeros(nr)
    He = np.zeros(nr)
    O = np.zeros(nr)
    Si = np.zeros(nr)
    for i in range(len(rho)):
        sum_ni = 0
        sum_h = 0
        sum_he = 0
        sum_o = 0
        sum_si =0
        for j in range(nth):
            for k in range(nph):
                sum_ni += Ni_file[k, j, i]
                sum_h += H_file[k, j, i]
                sum_he += He_file[k, j, i]
                sum_o += O_file[k, j, i]
                sum_si += Si_file[k, j, i]
        Ni[i] = sum_he / (nth * nph)
        H[i] = sum_h / (nth * nph)
        He[i] = sum_he / (nth * nph)
        O[i] = sum_o / (nth * nph)
        Si[i] = sum_si / (nth * nph)
    return Ni, H, He, O, Si

def rho_2D_r_th(filename):
    with h5py.File(str(filename), 'r') as f :
        rho_file = f['rho'][:, :, :]
    nr = int(rho_file.shape[2])
    nth = int(rho_file.shape[1])
    nph = int(rho_file.shape[0])
    rho = np.zeros((nr, nth))
    for i in range(nr):
        for j in range(nth):
            sum_rho = 0
            for k in range(nph):
                sum_rho += rho_file[k, j, i]
            rho[i] = sum_rho / nph
    return rho

def fgamma(filename):
    with h5py.File(str(filename), 'r') as f :
        gamma = f['gamma'][()]
    return gamma

def fiter(filename):
    with h5py.File(str(filename), 'r') as f :
        iter = f['iter'][()]
    return iter

def make_xc(x, n):
    dx = np.zeros(n)
    for i in range(2, n+2):
        dx[i-2] = x[i+1] - x[i]
    xc = np.zeros(n)
    for i in range(2, n+2):
        xc[i-2] = x[i] + dx[i-2] / 2
    xc_cgs = xc * 10**2
    return xc_cgs

def conversion_si_to_cgs(rho, u, P):
    rho_cgs = rho * 10**(-3)
    u_cgs = u * 10**2
    P_cgs = P * 10
    return rho_cgs, u_cgs, P_cgs

# ------------------------------------------

rho0, u0, P0, x0, t0 = read_start_file("../src/setups/v1d/v1d_1d_start.h5")
rhof, uf, Pf, Tf, xf, tf = read_file_1d_r("../src/setups/v1d/v1d_1e5.h5")
Nif, Hf, Hef, Of, Sif = read_file_1d_r_element("../src/setups/v1d/v1d_1e5.h5")

gamma = fgamma(filename)
iter = fiter(filename)
rho, u, P, T, x, t = read_file_1d_r(filename)
Ni, H, He, O, Si = read_file_1d_r_element(filename)

if (ndim ==3):
    rho, u, P, T = read_3d_file_in_1d_angular_mean(filename)
    Ni, H, He, O, Si = read_3d_file_element_in_1d_angular_mean(filename)

tday = t / 3600 / 24

print("    ")
print(f"Final time = {t:.3e} s or {tday} days")
print(f"Iteration number = {iter:3e}")
print("    ")

# cgs units --------------------------------

xcf_cm = make_xc(xf, len(rhof))
xc_cm = make_xc(x, len(rho))
xc0_cm = make_xc(x0, len(rho0))

rho0_cgs, u0_cgs, P0_cgs = conversion_si_to_cgs(rho0, u0, P0)
rho_cgs, u_cgs, P_cgs = conversion_si_to_cgs(rho, u, P)
rhof_cgs, uf_cgs, Pf_cgs = conversion_si_to_cgs(rhof, uf, Pf)

# ------------------------------------------

plt.figure(figsize=(12,8))
plt.suptitle(f'Loglog graph for v1d 1e5, t = {t:.1e} s ({tday:1f} jours)')
plt.subplot(221)
plt.plot(xc0_cm, rho0_cgs, "--", label=f"$t_0$= {t0:.1e} s")
plt.plot(xcf_cm, rhof_cgs, "--", color="red", label=f'$t_f$ = {tf:.1e} s')
plt.plot(xc_cm, rho_cgs, color='green', label=f'$t$ = {t:.1e} s')
plt.xlabel('rc (cm)'); plt.ylabel('Density ($g.cm^{-3}$)')
plt.xscale('log'); plt.yscale('log')
plt.grid()
plt.legend()

plt.subplot(222)
plt.plot(xc0_cm, u0_cgs, "--", label=f"$t_i$= {t0:.1e} s")
plt.plot(xcf_cm, uf_cgs, "--", color="red", label=f'$t_f$ = {tf:.1e} s')
plt.plot(xc_cm, u_cgs, color='green', label=f'$t$ = {t:.1e} s')
plt.xlabel('rc (cm)'); plt.ylabel('Velocity ($cm.s^{-1}$)')
plt.xscale('log')#; plt.yscale('log')
plt.grid()
plt.legend()

plt.subplot(223)
plt.plot(xc0_cm, P0_cgs, "--", label=f"$t_i$= {t0:.1e} s")
plt.plot(xcf_cm, Pf_cgs, "--", color="red", label=f'$t_f$ = {tf:.1e} s')
plt.plot(xc_cm, P_cgs, color='green', label=f'$t$ = {t:.1e} s')
plt.xlabel('rc (cm)'); plt.ylabel('Pressure ($g.cm^{-1}.s^{-2}$)')
plt.xscale('log'); plt.yscale('log')
plt.grid()
plt.legend()

plt.subplot(224)
#plt.plot(xc0_cm, T0, "--", label=f"$t_i$= {t0:.1e} s")
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
plt.grid()

plt.figure(figsize=(12,8))
plt.imshow(rho_r_th, origin='lower')
plt.colorbar() """

plt.figure()
plt.imshow(rho_2D_r_th(filename), aspect='auto', origin='lower', )
plt.colorbar()

plt.show()