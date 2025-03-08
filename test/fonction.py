import numpy as np
import h5py
import pandas as pd
import re

# ------------------------------------------------------------------------------
# Simulation parameters --------------------------------------------------------
# ------------------------------------------------------------------------------

def t_file(filename):
    """ Time of the simulation
    input    :
    filename : str : path of the file

    output :
    t      : float : current time
    """
    with h5py.File(str(filename), 'r') as f :
        t = f['current_time'][()]
    return t

def iter_file(filename):
    """ Number of iteration
    input    :
    filename : str : npath of the file

    output :
    iter   : int : current iteration
    """
    with h5py.File(str(filename), 'r') as f :
        iter = f['iter'][()]
    return iter

def gamma_file(filename):
    """ Number of iteration
    input    :
    filename : str : npath of the file

    output :
    gamma  : float : adiabatic index
    """
    with h5py.File(str(filename), 'r') as f :
        gamma = f['gamma'][()]
    return gamma

# ------------------------------------------------------------------------------
# Grid -------------------------------------------------------------------------
# ------------------------------------------------------------------------------

def size(filename):
    """ Size of the physical array
    input    :
    filename : str : npath of the file

    output :
    nx     : int : size of the first dimension
    ny     : int : size of the second dimension
    nz     : int : size of the third dimension
    """
    with h5py.File(str(filename), 'r') as f :
        rho_file = f['rho'][:, :, :]
        nx = int(rho_file.shape[2])
        ny = int(rho_file.shape[1])
        nz = int(rho_file.shape[0])
    return nx, ny, nz

def make_xc(x, n):
    """ Make x center
    intput :
    x      : array : node position
    n      : int   : size

    output :
    xc     : array : node center
    """
    dx = np.zeros(n)
    for i in range(2, n+2):
        dx[i-2] = x[i+1] - x[i]
    xc = np.zeros(n)
    for i in range(2, n+2):
        xc[i-2] = x[i] + dx[i-2] / 2
    return xc

def masse_cgs(rho, r):
    """" Mass calculation, with M_{star} = 1.53 M_{sun}
    input :
    rho   : array : density in cgs
    r     : array : position in the center in cgs

    output :
    M_cgs  : array : mass in cgs
    """
    n = len(rho)
    M_cgs = np.zeros(n)
    M_sun = 2e30 * 10**3 #g
    M_star = 1.53 * M_sun

    M_cgs[0] = M_star# + 4 / 3 * np.pi * rho[0] * (r[1]**3 - r[0]**3)
    for i in range(1, n):
        M_cgs[i] = M_cgs[i-1] + 4 / 3 * np.pi * rho[i] * (r[i]**3 - r[i-1]**3)

    return M_cgs

# ------------------------------------------------------------------------------
# cgs units --------------------------------------------------------------------
# ------------------------------------------------------------------------------

class Conversion_SI_to_cgs:
    """ Coversion from SI to cgs
    intput :
    data   : array or float : physical value in SI

    output:
    data   : array or float : physical value in cgs
    """
    def __init__(self, data):
        self.data = data

    def rho_cgs(self):
        return self.data * 10**(-3)

    def u_cgs(self):
        return self.data * 10**2

    def P_cgs(self):
        return self.data * 10

    def x_cgs(self):
        return self.data * 10**2

