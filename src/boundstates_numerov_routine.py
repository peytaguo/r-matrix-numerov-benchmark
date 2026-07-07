import numpy as np

from numerov import numerov_integrator, is_eval_checker, normalize_wavefunction

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