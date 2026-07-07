import numpy as np
import matplotlib.pyplot as plt

from numerov import HBAR_C, g, updater

#returns value of wavefunction on radial grid
def radial_numerov_s_wave(radius, divisions, reduced_mass, potential, energy):

    """
    Integrate the radial Schrödinger equation (s wave) using the Numerov method.

    Parameters
    ----------
    length : float
        Full-width of the integration (radial) interval [0, length].
    divisions : int
        Number of grid intervals.
    """

    step_size = radius / divisions
    h2_12 = step_size * step_size / 12

    #Assume the initial conditions:
    u_0 = 0
    u_1 = step_size 

    grids = np.linspace(0, radius, divisions + 1)
    
    approximation = np.zeros(divisions + 1)
    approximation[0] = u_0
    approximation[1] = u_1

    gvals = np.array([g(x, reduced_mass, potential, energy) for x in grids])

    
    for i in range(2, divisions+1):
            approximation[i] = updater(
                gvals[i],
                gvals[i-1], 
                gvals[i-2], 
                approximation[i-1], 
                approximation[i-2], 
                h2_12
            )
    return [grids, approximation, step_size]

def wrap_to_pi(delta):
    return (delta + np.pi) % (2 * np.pi) - np.pi


def phase_shift_s_wave(numerov_results, reduced_mass, energy, match_index=-2):
    """
    Extract neutral s-wave phase shift from

        u(r) ~ A sin(kr + delta)

    at a large matching radius.
    """

    grids = np.array(numerov_results[0])
    u = np.array(numerov_results[1])
    step_size = numerov_results[2]

    r_match = grids[match_index]
    u_match = u[match_index]

    du_match = (u[match_index + 1] - u[match_index - 1]) / (2 * step_size)

    k = np.sqrt(2 * reduced_mass * energy) / HBAR_C

    delta = np.arctan2(k * u_match, du_match) - k * r_match

    return wrap_to_pi(delta)


def routine(radius, divisions, reduced_mass, potential, energy, match_index=-2):
    numerov_results = radial_numerov_s_wave(
        radius,
        divisions,
        reduced_mass,
        potential,
        energy
    )

    delta = phase_shift_s_wave(
        numerov_results,
        reduced_mass,
        energy,
        match_index=match_index
    )

    return numerov_results, delta

def phase_shift_curve(radius, divisions, reduced_mass, potential, energies, match_index=-2):
    deltas = []

    for energy in energies:
        numerov_results = radial_numerov_s_wave(
            radius=radius,
            divisions=divisions,
            reduced_mass=reduced_mass,
            potential=potential,
            energy=energy
        )

        delta = phase_shift_s_wave(
            numerov_results,
            reduced_mass=reduced_mass,
            energy=energy,
            match_index=match_index
        )

        deltas.append(delta)

    return np.array(deltas)

def V_free(r):
    return 0.0


mass = 469.136
radius = 30.0
divisions = 5000

energies = np.linspace(1.0, 30.0, 100)

deltas = phase_shift_curve(
    radius=radius,
    divisions=divisions,
    reduced_mass=mass,
    potential=V_free,
    energies=energies
)

plt.close("all")

plt.figure(figsize=(8, 5))
plt.plot(energies, deltas, label=r"$\delta_0(E)$")
plt.axhline(0, linestyle="--", linewidth=1)
plt.xlabel("Energy / MeV")
plt.ylabel(r"$\delta_0$ / rad")
plt.title("Free s-wave phase shift")
plt.grid(True)
plt.legend()
plt.show()

V0 = 30.0
R = 3.0

def V_gaussian(r):
    return -V0 * np.exp(-(r / R)**2)


energies = np.linspace(1.0, 30.0, 150)

deltas = phase_shift_curve(
    radius=40.0,
    divisions=8000,
    reduced_mass=mass,
    potential=V_gaussian,
    energies=energies
)

# Optional: unwrap so the plotted curve does not jump at +/- pi.
deltas_unwrapped = np.unwrap(deltas)

plt.close("all")

plt.figure(figsize=(8, 5))
plt.plot(energies, deltas_unwrapped, label=r"$\delta_0(E)$")
plt.axhline(0, linestyle="--", linewidth=1)
plt.xlabel("Energy / MeV")
plt.ylabel(r"$\delta_0$ / rad")
plt.title("s-wave phase shift for Gaussian attractive well")
plt.grid(True)
plt.legend()
plt.show()