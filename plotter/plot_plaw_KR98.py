import numpy as np
import matplotlib.pyplot as plt
from enseisro.synthetics import create_rot_prof_KR98 as create_rot_prof

Nstars = 100

# creating the range of Prot we want to use
Prot = np.logspace(np.log10(1), np.log10(28), Nstars)

# Delta Omega / Omega
DOmega_by_Omega_true = np.logspace(np.log10(5e-4), np.log10(0.015), Nstars)

# getting the Delta Omega and Omega in nHz
DOmega_by_Omega_gen, Omega = create_rot_prof.get_DeltaOmega_from_Prot(Prot, 'G2')
    
# getting the DOmega in nHz
DOmega = DOmega_by_Omega_gen * Omega

print('Omega in nHz:\n', Omega)
print('DOmega_by_Omega_gen:\n', DOmega_by_Omega_gen)
print('Domega in nHz:\n', DOmega)  
'''
print('Rotation periods in days:\n', Prot)
print('Omega in nHz:\n', Omega)
print('Delta Omega in nHz:\n', Delta_Omega)
print('Delta Omega / Omega:\n', Delta_Omega/Omega)
'''
    
plt.figure()
    
plt.loglog(Omega, DOmega)
# plt.loglog(Prot, DOmega_by_Omega_gen, 'x')
plt.xlabel('$\Omega$ in nHz')
plt.ylabel('$\Delta \Omega$ in nHz')

plt.savefig('KR98.pdf')
