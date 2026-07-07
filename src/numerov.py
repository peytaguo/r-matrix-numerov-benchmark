import numpy as np

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


#returns value of wavefunction on grid
def numerov_integrator(length, divisions, psi_0, psi_1, mass, potential, energy, full_integration):
    """
    Integrate the 1D Schrödinger equation using the Numerov method.

    Parameters
    ----------
    length : float
        Half-width of the integration interval [-length, length].
    divisions : int
        Number of grid intervals.
    psi_0, psi_1 : float
        Wavefunction values at the first two grid points.
    """
    step_size = 2 * length / divisions
    h2_12 = step_size * step_size / 12

    grids = np.linspace(-length, length, divisions + 1)
    
    approximation = np.zeros(divisions + 1)
    approximation[0] = psi_0
    approximation[1] = psi_1

    gvals = np.array([g(x, mass, potential, energy) for x in grids])

    if divisions % 2 != 0:
        raise ValueError("divisions must be even so that x=0 is a grid point")

    if full_integration:
        effective_division = divisions+1
    else:
        # one past midpoint: 
        effective_division = divisions // 2 + 1
    
    for i in range(2, effective_division+1):
            approximation[i] = updater(
                gvals[i],
                gvals[i-1], 
                gvals[i-2], 
                approximation[i-1], 
                approximation[i-2], 
                h2_12
            )

    return [grids, approximation, full_integration, effective_division, step_size]


def is_eval_checker(numerov_results):

    tolerance = 1

    approximation = numerov_results[1]
    effective_division = numerov_results[3]
    last_three = approximation[effective_division-2:effective_division+1]
    step_size = numerov_results[4]

    psi_0_approx = last_three[1]
    dpsi_0_approx = (last_three[-1] - last_three[0]) / (2 * step_size)
    criterion = 2 * dpsi_0_approx * psi_0_approx 

    if abs( criterion ) <= tolerance:
        return [True, 0]
    else:
        return [False, criterion]

def routine():
    pass

#test_1 = numerov_integrator(1, 5, 1, 2)
#print(test_1)