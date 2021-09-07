import numpy as np
from .constants import *

__all__ = ["mass_to_rad",
           "rad_to_mass"]


def mass_to_rad(mp): 
    '''
    mp - float 
    
    Uses the mass-radius relation given in Chen and Kipping (2017).  This method takes mass and 
    outputs radius.  mp is given in Earth masses, and rp is returned in Earth radii.  The specific numbers were obtained
    from here: https://exoplanetarchive.ipac.caltech.edu/docs/pscp_calc.html.
    '''
    M = np.log10(mp)
    
    if mp < 2.04:                                          
        C, S = 0.00346, 0.2790
    elif mp < 132:
        C, S = -0.0925, 0.589
    elif mp < 26600:
        C, S = 1.25, -0.044
    else:
        C, S = -2.85, 0.881
        
    return 10**(C + M*S)


def rad_to_mass(rp):
    '''
    rp - float  
    
    Takes a radius measurement and outputs its corresponding mass using the mass-radius relation in Chen and Kipping
    (2017).  This is kind of like the inverse of mass_radius above.  The specific numbers were obtained from here: 
    https://exoplanetarchive.ipac.caltech.edu/docs/pscp_calc.html
    '''
    R = np.log10(rp)
    
    if rp < 1.23:                                         
        C, S = 0.00346, 0.2790
    elif rp < 11.1:
        C, S = -0.0925, 0.589
    elif rp < 14.3:
        C, S = np.nan, np.nan
    else:
        C, S = -2.85, 0.881
        
    return 10**((R-C)/S)
