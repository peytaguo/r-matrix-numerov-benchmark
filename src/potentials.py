import math 

# Narrow resonance 12C+p example

E2 = 1.43996448  #units MeV fm

#r in fm units
#energies in MeV units. 

def nuclear_potential(r):
    return -73.8*math.exp(-(r/2.70)**2)

def coulomb_potential(r):
    return 6*E2/(r)