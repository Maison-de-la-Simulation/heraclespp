import sys
import h5py
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import rc

R_sun_cgs = 6.957 * 10**10

print("********************************")
print("             V1D")
print("********************************")

rc('figure')
plt.rc('font', family='serif', serif='Palatino', size=13)
rc('axes', linewidth=2)
rc("text", usetex=True)
rc('legend', fontsize=10)
rc('xtick.major', size=5, width=2)
rc('ytick.major', size=5, width=2)
rc('xtick.minor', size=3, width=1)
rc('ytick.minor', size=3, width=1)

# ------------------------------------------------------------------------------

filename = sys.argv[1]
#ndim = input("Dimension of the simulation (1 or 3): ")

# ------------------------------------------------------------------------------

def read_start_file(filename):
    with h5py.File(str(filename), 'r') as f:
        rho = f['rho_1d'][()]
        u = f['u_1d'][()]
        P = f['P_1d'][()]
        x = f['x'][()]
        t = f['current_time'][()]
    return rho, u, P, x, t

def read_file_1d_r(filename):
    with h5py.File(str(filename), 'r') as f:
        rho = f['rho'][0, 0, :]
        u = f['ux'][0, 0, :]
        P = f['P'][0, 0, :]
        T = f['T'][0, 0, :]
        E = f['T'][0, 0, :]
        x = f['x'][()]
        t = f['current_time'][()]
    return rho, u, P, T, E, x, t

def read_file_1d_r_element(filename):
    with h5py.File(str(filename), 'r') as f:
        Ni = f['fx0'][0, 0, :]
        H = f['fx1'][0, 0, :]
        He = f['fx2'][0, 0, :]
        O = f['fx3'][0, 0, :]
        Si = f['fx4'][0, 0, :]
    return Ni, H, He, O, Si

def fgamma(filename):
    with h5py.File(str(filename), 'r') as f:
        gamma = f['gamma'][()]
    return gamma

def fiter(filename):
    with h5py.File(str(filename), 'r') as f:
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
rhof, uf, Pf, Tf, Ef, xf, tf = read_file_1d_r("../src/setups/v1d/v1d_1e5.h5")

gamma = fgamma(filename)
iter = fiter(filename)
rho, u, P, T, E, x, t = read_file_1d_r(filename)
Ni, H, He, O, Si = read_file_1d_r_element(filename)

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
plt.plot(xc0_cm / R_sun_cgs, np.log10(rho0_cgs), "--", label=f"$t_0$= {t0:.1e} s")
plt.plot(xcf_cm / R_sun_cgs, np.log10(rhof_cgs), "--", color="red", label=f'$t_f$ = {tf:.1e} s')
plt.plot(xc_cm / R_sun_cgs, np.log10(rho_cgs), color='green', label=f'$t$ = {t:.1e} s')
plt.xlabel(r'$r / R_{\odot}$'); plt.ylabel(r'log($\rho$) [$g.cm^{-3}$]')
plt.legend()

plt.subplot(222)
plt.plot(xc0_cm / R_sun_cgs, u0_cgs / 10**8, "--", label=f"$t_i$= {t0:.1e} s")
plt.plot(xcf_cm / R_sun_cgs, uf_cgs / 10**8, "--", color="red", label=f'$t_f$ = {tf:.1e} s')
plt.plot(xc_cm / R_sun_cgs, u_cgs / 10**8, color='green', label=f'$t$ = {t:.1e} s')
#plt.axline((0, 1), slope=0, label=r"$u_{shift}$", color='blue')
plt.xlabel(r'$r / R_{\odot}$'); plt.ylabel(r'$u$ [$/ 10^8$ cm.s$^{-1}$]')
plt.legend()

plt.subplot(223)
plt.plot(xcf_cm / R_sun_cgs, np.log10(Tf / 10**6), "--", color="red", label=f'$t_f$ = {tf:.1e} s')
plt.plot(xc_cm / R_sun_cgs, np.log10(T / 10**6), color='green', label=f'$t$ = {t:.1e} s')
plt.xlabel(r'$r / R_{\odot}$'); plt.ylabel(r'log($T / 10^6$) [K]')
plt.legend()

plt.subplot(224)
plt.plot(xc_cm / R_sun_cgs, Ni, c='red', label="Ni56")
plt.plot(xc_cm / R_sun_cgs, H, c="steelblue",label="H")
plt.plot(xc_cm / R_sun_cgs, He, c="plum", label="He")
plt.plot(xc_cm / R_sun_cgs, O, c='green', label="O")
plt.plot(xc_cm / R_sun_cgs, Si, color="gold",label="Si")
plt.xlabel(r'$r / R_{\odot}$'); plt.ylabel(r'X')
plt.legend()

plt.show()
