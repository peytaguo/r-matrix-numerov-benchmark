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

    if divisions % 2 != 0:
        raise ValueError("divisions must be even so that x=0 is a grid point")
    
    if divisions < 2:
        raise ValueError("divisions must be at least 2")

    step_size = 2 * length / divisions
    h2_12 = step_size * step_size / 12

    grids = np.linspace(-length, length, divisions + 1)
    
    approximation = [0 for i in range(divisions + 1)]
    approximation[0] = psi_0
    approximation[1] = psi_1

    gvals = np.array([g(x, mass, potential, energy) for x in grids])

    if full_integration:
        effective_division = divisions
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

    return [grids, approximation[:effective_division + 1], full_integration, effective_division, step_size]


def is_eval_checker(numerov_results):

    approximation = numerov_results[1]
    effective_division = numerov_results[3]
    last_three = approximation[effective_division-2:effective_division+1]
    step_size = numerov_results[4]

    psi_0_approx = last_three[1]
    dpsi_0_approx = (last_three[-1] - last_three[0]) / (2 * step_size)
    criterion = 2 * dpsi_0_approx * psi_0_approx 

    return [criterion, dpsi_0_approx]

def routine(initial_guess, expected_spacing, iterations, length, divisions, psi_0, psi_1, mass, potential, tolerance):
    error_lst = []

    numerov_results = numerov_integrator(
        length, 
        divisions,
        psi_0,
        psi_1,
        mass,
        potential,
        initial_guess,
        full_integration=False)
    
    error = is_eval_checker(numerov_results)
    error_lst.append(error)

    initial_guess += expected_spacing

    numerov_results = numerov_integrator(
        length, 
        divisions,
        psi_0,
        psi_1,
        mass,
        potential,
        initial_guess,
        full_integration=False)
    
    error = is_eval_checker(numerov_results)
    error_lst.append(error)

    for i in range(iterations):
        sign = error_lst[-2][0] * error_lst[-1][0]
        if sign >= 0:
            expected_spacing /= 2
        elif sign < 0:
            expected_spacing /= -2
        
        initial_guess += expected_spacing
        
        numerov_results = numerov_integrator(
        length, 
        divisions,
        psi_0,
        psi_1,
        mass,
        potential,
        initial_guess,
        full_integration=False)

        error = is_eval_checker(numerov_results)
        error_lst.append(error)

    if abs(error_lst[-1][0]) > tolerance: 
        raise ValueError("Increase iterations or adjust initial_guess / expected_spacing.")

    if abs(error_lst[-1][-1]) < tolerance:
        alpha = 1
    else:
        alpha = -1
        
    full_approximation = reflect_solution(numerov_results, divisions, alpha)
    full_approximation = normalize_wavefunction(numerov_results[0], full_approximation)

    print("final error = {}".format(error_lst[-1][0]))
    return [numerov_results[0], full_approximation]


def reflect_solution(numerov_results, divisions, alpha):
    half_approximation = numerov_results[1]
    midpoint = divisions // 2

    full_approximation = [0 for _ in range(divisions + 1)]

    # Copy from -L to 0.
    for i in range(midpoint + 1):
        full_approximation[i] = half_approximation[i]

    # Reflect from the left side onto the right side.
    for i in range(midpoint):
        full_approximation[divisions - i] = alpha * half_approximation[i]

    return full_approximation

def normalize_wavefunction(grid, psi):
    psi = np.array(psi, dtype=float)
    norm = np.sqrt(np.trapezoid(psi**2, grid))

    if norm == 0:
        raise ValueError("Cannot normalize wavefunction with zero norm.")

    return psi / norm

#Harmonic Oscillator Test Case

mass = 938.272  # MeV/c^2
a = 2.0         # oscillator length in fm

E0_exact = 0.5 * HBAR_C**2 / (mass * a**2)

length = 10.0
divisions = 5000
h = 2 * length / divisions


def V_ho(x):
    return HBAR_C**2 * x**2 / (2 * mass * a**4)

"""

def psi_0_exact(x):
    return np.exp(-x**2 / (2 * a**2))

psi_left = psi_0_exact(-length)
psi_next = psi_0_exact(-length + h)

results = routine(
    initial_guess=E0_exact-0.5, 
    expected_spacing=0.1, 
    iterations=3, 
    length=length, 
    divisions=divisions, 
    psi_0=psi_left, 
    psi_1=psi_next,
    mass=mass,
    potential=V_ho,
    tolerance=10
    )

grid = results[0]
psi_num = np.array(results[1])
psi_ex = psi_0_exact(grid)
psi_exact_n = normalize_wavefunction(grid, psi_ex)

max_error = np.max(np.abs(psi_num - psi_ex))

plt.figure(figsize=(8, 5))
plt.plot(grid, psi_exact_n, label="Exact ground state")
plt.plot(grid, psi_num, "--", label="Numerov")
plt.xlabel("x / fm")
plt.ylabel(r"$\psi(x)$")
plt.title(f"Harmonic oscillator ground state, E0 = {E0_exact:.6f} MeV\nmax |error| = {max_error:.2e}")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"Exact ground-state energy: {E0_exact:.8f} MeV")
print(f"Maximum absolute wavefunction error after normalisation: {max_error:.3e}")

"""


def psi_1_exact(x):
    # Unnormalised first excited-state wavefunction
    return x * np.exp(-x**2 / (2 * a**2))

E0_exact = 0.5 * HBAR_C**2 / (mass * a**2)
E1_exact = 3 * E0_exact

psi_left = psi_1_exact(-length)
psi_next = psi_1_exact(-length + h)

results = routine(
    initial_guess=E1_exact-0.5, 
    expected_spacing=0.0001, 
    iterations=5000, 
    length=length, 
    divisions=divisions, 
    psi_0=psi_left, 
    psi_1=psi_next,
    mass=mass,
    potential=V_ho,
    tolerance=10e-1
    )

grid = results[0]
psi_num = np.array(results[1])
psi_exact = psi_1_exact(grid)

psi_num_n = normalize_wavefunction(grid, psi_num)
psi_exact_n = normalize_wavefunction(grid, psi_exact)

if np.trapezoid(psi_num_n * psi_exact_n, grid) < 0:
    psi_num_n *= -1

max_error = np.max(np.abs(psi_num_n - psi_exact_n))

plt.figure(figsize=(8, 5))
plt.plot(grid, psi_exact_n, label="Exact first excited state")
plt.plot(grid, psi_num_n, "--", label="Numerov")
plt.xlabel("x / fm")
plt.ylabel(r"$\psi(x)$")
plt.title(f"Harmonic oscillator first excited state, E1 = {E1_exact:.6f} MeV\nmax |error| = {max_error:.2e}")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()