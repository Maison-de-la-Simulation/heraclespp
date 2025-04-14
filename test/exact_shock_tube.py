#           SHOCK TUBE
#           Exact solution
#
# from Toro

import numpy as np


def CI(x, inter, tabl, tabr):
    """Shock tube initialisation

    input :
    x     : array : position
    inter : float : interface position
    tabl  : array : density, velocity, pressure and speed sound left
    tabr  : array : right

    output :
    rho0  : array : density t=0
    u0    : array : velocity t=0
    P0    : array : pressure t=0
    """
    rhol, ul, Pl, cl = tabl
    rhor, ur, Pr, cr = tabr

    rho0 = np.zeros(len(x))
    u0 = np.zeros(len(x))
    P0 = np.zeros(len(x))

    for i in range(len(x)):
        if x[i] <= inter:
            rho0[i] = rhol
            u0[i] = ul
            P0[i] = Pl

        else:
            rho0[i] = rhor
            u0[i] = ur
            P0[i] = Pr

    return rho0, u0, P0


def StarPU(tabl, tabr, gamma):
    """Pressure and velocity in the star region

    input :
    tabl  : array : density, velocity, pressure and speed sound left
    tabr  : array : right
    gamma : float : adiabtic constant

    output  :
    P_star : float : pressure
    u_star : float : speed
    """
    rhol, ul, Pl, cl = tabl
    rhor, ur, Pr, cr = tabr

    tolpre = 1e-6  # Variation threshold
    nriter = 20
    P_start = GuessP(tabl, tabr, gamma)  # initale pressure
    P_old = P_start
    u_diff = ur - ul

    for i in range(nriter):
        fr, frd = Prefun(P_old, Pr, rhor, cr, gamma)
        fl, fld = Prefun(P_old, Pl, rhol, cl, gamma)
        P_star = P_old - (fl + fr + u_diff) / (fld + frd)
        change = 2 * (P_star - P_old) / (P_star + P_old)

        if change < tolpre:
            i = i + 1

        if P_star < tolpre:
            P_old = P_star
            break

    u_star = 0.5 * (ul + ur + fr - fl)
    return u_star, P_star


def GuessP(tabl, tabr, gamma):
    """Define an intelligent initial pressure in the star region

    input :
    tabl  : array : density, velocity, pressure and speed sound left
    tabr  : array : right
    gamma : float :adiabatic constant

    output  :
    P_star : float : pressure in the star region
    """
    z = (gamma - 1) / (2 * gamma)
    g4 = 2 / (gamma - 1)
    g5 = 2 / (gamma + 1)
    g6 = (gamma - 1) / (gamma + 1)
    g7 = (gamma - 1) / 2

    rhol, ul, Pl, cl = tabl
    rhor, ur, Pr, cr = tabr

    quser = 2
    rho_star = (1 / 2) * (rhol + rhor)
    c_star = (1 / 2) * (cl + cr)
    P_pvrs = (1 / 2) * (Pl + Pr) + (1 / 2) * (ul - ur) * rho_star * c_star
    P_pvrs = np.max([0, P_pvrs])
    Pmin = np.min([Pl, Pr])
    Pmax = np.max([Pl, Pr])
    qmax = Pmax / Pmin

    if (qmax < quser) and ((Pmin <= P_pvrs) and (P_pvrs <= Pmax)):
        P_star = P_pvrs
    else:
        if P_pvrs < Pmin:
            P_lr = (Pl / Pr) ** z
            u_star = (P_lr * ul / cl + ur / cr + g4 * (P_lr - 1)) / (
                P_lr / cl + 1 / cr
            )  # Vitesse moyenne
            Ptl = 1 + g7 * (ul - u_star) / cl
            Ptr = 1 + g7 * (u_star - ur) / cr
            P_star = (1 / 2) * (Pl * Ptr ** (1 / z) + Pr * Ptr ** (1 / z))
        else:
            gl = np.sqrt((g5 / rhol) / (P_pvrs + g6 * Pl))
            gr = np.sqrt((g5 / rhor) / (P_pvrs + g6 * Pr))
            P_star = (gl * Pl + gr * Pr - (ur - ul)) / (gl + gr)

    return P_star


def Prefun(P, Pk, rhok, ck, gamma):
    """Evaluate the pressure function

    input :
    P     : float : input estimate pressure
    Pk    : float : pressure left k=l or right k=r
    rhok  : float : density left or right
    ck    : float : sound speed left or right
    gamma : float :abiabatic constant

    output :
    fk    : float : pressure function left or right
    fd    : float : pressure function derivate
    """
    z = (gamma - 1) / (2 * gamma)
    g2 = (gamma + 1) / (2 * gamma)
    g4 = 2 / (gamma - 1)
    g5 = 2 / (gamma + 1)
    g6 = (gamma - 1) / (gamma + 1)

    if P < Pk:  # Rarefaction wave
        P_rat = P / Pk
        fk = g4 * ck * (P_rat**z - 1)
        fd = (1 / (rhok * ck)) * P_rat ** (-g2)
    else:  # Shock wave
        Ak = g5 / rhok
        Bk = g6 * Pk
        qrt = np.sqrt(Ak / (P + Bk))
        fk = (P - Pk) * qrt
        fd = (1 - 0.5 * (P - Pk) / (Bk + P)) * qrt

    return fk, fd


def Sample(P_star, u_star, S, tabl, tabr, gamma):
    """Find solution between the wave

    input  :
    P_star : float : pressure in the star region
    u_star : float : velocity in the star region
    S      : float : velocity depends of the position
    tabl   : array : density, velocity, pressure and speed sound left
    tabr   : array : right
    gamma  : float : adiabatic constant

    output :
    rho   : float : density
    u     : float : speed
    P     : float : pressure
    """
    z = (gamma - 1) / (2 * gamma)
    g2 = (gamma + 1) / (2 * gamma)
    g3 = 2 * gamma / (gamma - 1)
    g4 = 2 / (gamma - 1)
    g5 = 2 / (gamma + 1)
    g6 = (gamma - 1) / (gamma + 1)
    g7 = (gamma - 1) / 2

    rhol, ul, Pl, cl = tabl
    rhor, ur, Pr, cr = tabr
    if S < u_star:  # Left of the discontinuity
        if P_star <= Pl:  # Rearefaction wave
            Shl = ul - cl
            if S <= Shl:
                rho = rhol
                u = ul
                P = Pl
            else:
                cl_star = cl * (P_star / Pl) ** (z)
                Stl = u_star - cl_star
                if S > Stl:
                    rho = rhol * (P_star / Pl) ** (1 / gamma)
                    u = u_star
                    P = P_star
                else:
                    aide = g5 * (cl + g7 * (ul - S))
                    rho = rhol * (aide / cl) ** g4
                    u = g5 * (cl + g7 * ul + S)
                    P = Pl * (aide / cl) ** g3
        else:  # Shock wave
            rapl = P_star / Pl
            Sl = ul - cl * np.sqrt(g2 * rapl + z)
            if S < Sl:
                rho = rhol
                u = ul
                P = Pl
            else:
                rho = rhol * (rapl + g6) / (rapl * g6 + 1)
                u = u_star
                P = P_star
    else:  # Right of the discontinuity
        if P_star > Pr:  # Shock wave
            rapr = P_star / Pr
            Sr = ur + cr * np.sqrt(g2 * rapr + z)
            if S > Sr:
                rho = rhor
                u = ur
                P = Pr
            else:
                rho = rhor * (rapr + g6) / (rapr * g6 + 1)
                u = u_star
                P = P_star
        else:  # Rarefaction wave
            Shr = ur + cr
            if S >= Shr:
                rho = rhor
                u = ur
                P = Pr
            else:
                cr_star = cr * (P_star / Pr) ** (z)
                Str = u_star + cr_star
                if S <= Str:
                    rho = rhor * (P_star / Pr) ** (1 / gamma)
                    u = u_star
                    P = P_star
                else:
                    aide2 = g5 * (cr - g7 * (ur - S))
                    rho = rhor * (aide2 / cr) ** g4
                    u = g5 * (-cr + g7 * ur + S)
                    P = Pr * (aide2 / cr) ** g3
    return rho, u, P


def ExactShockTube(x, inter, var0L, var0R, t, gamma):
    """Exact solution

    input :
    x     : array : position
    inter : float : interface position
    var0L : array : density, velocity, pressure and speed sound left
    var0R : array : right
    t     : float : output time
    gamma  : float : adiabatic constant

    output :
    rho   : array : density
    u     : array : speed
    P     : array : pressure
    e     : array : internal energy
    """
    rho = np.zeros(len(x))
    u = np.zeros(len(x))
    P = np.zeros(len(x))
    u_star, P_star = StarPU(var0L, var0R, gamma)
    for i in range(len(x)):
        S = (x[i] - inter) / t
        rho[i], u[i], P[i] = Sample(P_star, u_star, S, var0L, var0R, gamma)
    e = P / (rho * (gamma - 1))
    return rho, u, P, e
