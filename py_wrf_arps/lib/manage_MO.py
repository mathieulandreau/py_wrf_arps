import numpy as np
import matplotlib.pyplot as plt

from ..lib import constants

def MO_phim(z_L):
    """Stability function phi_m for unstable conditions (Businger-Dyer)
    """
    if z_L <= 1e-5 :
        return (1 - 16 * z_L) ** (-0.25)
    else :
        raise(Exception("This is not a convective profile, please add the stable similarity function"))

def MO_length(u_star, T_star):
    """Monin-Obukhov length
    """
    return (u_star**2)/(constants.KARMAN*constant.BETA*T_star)
        
def MO_velocity(u_star, T_star, z0, Z):
    """Velocity profile in the Monin-Obukhov ST
    """
    L = MO_length(u_star, T_star)
    return (u_star / constants.KARMAN) * np.log(Z / z0) / phi_m(Z / L)
