#re-usable python code for numerov integration

import numpy as np
import matplotlib.pyplot as plt

# constants

HBAR_C = 197.3269804 #MeV fm

#mass with units MeV/c^2 
def g(x, mass, potential, energy):

    return 2 * mass * (energy - potential(x)) / HBAR_C**2 


def updater(g_3, g_2, g_1, psi_2, psi_1, h2_12):

    psi_c_numerator_a = 2 * (1 - 5 * h2_12 * g_2) * psi_2
    psi_c_numerator_b = (1 + h2_12 * g_1) * psi_1
    psi_c_denominator = 1 + h2_12 * g_3
    return (psi_c_numerator_a - psi_c_numerator_b) / psi_c_denominator

def normalize_wavefunction(grid, psi):
    psi = np.array(psi, dtype=float)
    norm = np.sqrt(np.trapezoid(psi**2, grid))

    if norm == 0:
        raise ValueError("Cannot normalize wavefunction with zero norm.")

    return psi / norm