import numpy as np

import matplotlib.pyplot as plt 

from numerov import numerov_integrator, is_eval_checker, normalize_wavefunction, HBAR_C
from boundstates_numerov_routine import routine

#Harmonic Oscillator Test Case

mass = 938.272  # MeV/c^2
a = 2.0         # oscillator length in fm

E0_exact = 0.5 * HBAR_C**2 / (mass * a**2)

length = 10.0
divisions = 5000
h = 2 * length / divisions


def V_ho(x):
    return HBAR_C**2 * x**2 / (2 * mass * a**4)


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