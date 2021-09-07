import numpy as np
import csv
import sys
import os
import scipy.optimize as op
from scipy import stats

from numpy.polynomial.polynomial import polyroots
from .constants import *

__all__ = ["read_csv_file",
           "get_csv_data",
           "get_num_vals_from_str",
           "mp_from_kamp",
           "kamp_from_mp",
           "extrapolated_anderson_2samp",
           "petrovich_stability"]


def read_csv_file(filename, k_index=0, v_index=1):
    """
    Read a csv file and return keys and values for use in a dictionary
    
    Parameters
    ----------
    filename : string
        csv file
    k_index : int
        index where keys start
    v_index: int
        index where values start
        
    Returns
    -------
        keys : list of keys
        values : list of values
    """
    data = []
    with open(filename) as infile:
        reader = csv.reader(infile)

        for row in reader:
            data.append(row)

        keys   = data[k_index]
        values = data[v_index:]

        return keys, values


    
def get_csv_data(keyname, keys, values):
    """
    Put the keys and values outputs of read_csv_file() into a useable format
    
    Parameters
    ----------
    keyname : string
        column definition
    keys : list
        keys
    values : list
        values corresponding to each key
    """
    kid = keys.index(keyname)
    
    outdata = []
    for row in values:
        outdata.append(row[kid])
    
    return outdata



def get_num_vals_from_str(dlist, dtype="float"):
    """
    Get numerical values from a list of data strings, replacing missing values with 'nan'
    
    arr : list
    """
    arr = np.array(dlist)
    arr[arr == ""] = "nan"
    
    return np.array(arr, dtype=dtype)




def mp_from_kamp(kamp, per, mstar, ecc=0.0):
    """
    Parameters
    ----------
        kamp : float
            RV semiamplitude [m/s]
        per : float
            orbital period [days]
        mstar : float
            stellar mass [M_sun]
        ecc : float (optional)
            eccentricity
            
    Returns:
        mp : float
            planet mass [M_earth]
    """
    # convert all quantities to SI units
    K = kamp*1.0
    P = per*24*3600
    M = mstar*MSUN
    G = BIGG
    
    roots = polyroots([-P*K**3, -2*P*K**3, -P*K**3, 2*pi*G*M/(1-ecc**2)**1.5])
    mpms = roots[np.isreal(roots)][0].real
    
    return mpms*M/MEARTH



def kamp_from_mp(per, mp, Ms, ecc=0.0):
    """
    See Lovis & Fischer 2010, Eq 14 in Exoplanets (Seager 2010)
    
    Parameters
    ----------
        per : float
            orbital period [days]
        mp : array-like
            planet mass [M_earth]
        mstar : float
            stellar mass [M_sun]
        ecc : float (optional)
            eccentricity
            
    Returns:
        kamp : float
            RV semiamplitude [m/s]
    """
    # convert all quantities to SI units
    mj = mp*MEARTH/MJUPITER
    P = per/365.24
    G = BIGG
    
    return 28.4329/np.sqrt(1-ecc**2) * mj * (Ms + mj*MJUPITER/MSUN)**(-2/3) * P**(-1/3)



def extrapolated_anderson_2samp(x1, x2):
    A, cv, p = stats.anderson_ksamp([x1, x2])

    alpha = np.array([0.25, 0.10, 0.05, 0.025, 0.01, 0.005, 0.001])
    
    res_fxn = lambda theta, x, y: y - 1/(1 + theta[0]*np.exp(x/theta[1]))
    
    fit, success = op.leastsq(res_fxn, [0.25, 1], args=(cv, alpha))
    
    z = np.linspace(-2,6)
    p_new = 1/(1 + fit[0]*np.exp(A/fit[1]))
    
    return A, p_new



def P_to_a(P, Mstar):
    """
    Convenience function to convert periods to semimajor axis from Kepler's Law
    
    P: orbital periods [days]
    Mstar: stellar mass [solar masses]
    """
    Pearth = 365.24    # [days]
    aearth = 215.05    # [solar radii]
    
    return aearth * ((P/Pearth)**2 *(1/Mstar))**(1/3)



def petrovich_stability(per, mp, Ms, ecc=np.zeros(2)):
    """
    Parameters
    ----------
        per : ndarray
            orbital periods [days]
        mp : ndarray
            planet masses [M_earth]
        Ms : float
            stellar mass [M_sun]
        ecc : ndarray (optional)
            eccentricities
    """
    sma = P_to_a(per, Ms)
    
    left = (sma[1]*(1-ecc[1]))/(sma[0]*(1+ecc[0]))
    right = 2.4*np.max(mp/Ms*MEARTH/MSUN)**(1/3) * (sma[1]/sma[0])**(1/2) + 1.15
    
    return left > right