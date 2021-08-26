import numpy as np
import matplotlib.pyplot as plt
from enseisro.synthetics import create_rot_prof_KR98 as create_rot_prof

Nstars = 100

################## Plotting the RDR ##################

# creating the range of Prot we want to use
Prot = np.logspace(np.log10(1), np.log10(28), Nstars)

# Delta Omega / Omega
DOmega_by_Omega_true_RDR = np.logspace(np.log10(5e-4), np.log10(0.015), Nstars)

# getting the Delta Omega and Omega in nHz
DOmega_by_Omega_RDR_G2, DOmega_by_Omega_LDR_G2, Omega = create_rot_prof.get_DeltaOmega_from_Prot(Prot, 'G2')
DOmega_by_Omega_RDR_K5, DOmega_by_Omega_LDR_K5, __ = create_rot_prof.get_DeltaOmega_from_Prot(Prot, 'K5')

'''    
# getting the DOmega in nHz
DOmega_RDR = DOmega_by_Omega_RDR * Omega

print('Omega in nHz:\n', Omega)
print('DOmega_by_Omega_gen:\n', DOmega_by_Omega_RDR)
print('Domega in nHz:\n', DOmega_RDR)  

print('Rotation periods in days:\n', Prot)
print('Omega in nHz:\n', Omega)
print('Delta Omega RDR in nHz:\n', Delta_Omega_RDR)
print('Delta Omega RDR / Omega:\n', Delta_Omega_RDR/Omega)
'''
    
# plt.figure()
fig, ax = plt.subplots(1,2)

ax[0].loglog(Prot, DOmega_by_Omega_LDR_G2, 'k', label='G2')
ax[0].loglog(Prot, DOmega_by_Omega_LDR_K5, 'r', label='K5')
# plt.loglog(Prot, DOmega_by_Omega_LDR, 'x')
ax[0].set_xlabel('Rotation period (day)')
ax[0].set_ylabel('Differential rotation in nHz')
ax[0].set_title('LDR')
ax[0].legend()

# plt.savefig('KR98_RDR.pdf')

################## Plotting the LDR ##################

# Delta Omega / Omega for LDR y-axis
DOmega_by_Omega_true_LDR = np.logspace(np.log10(6e-3), np.log10(0.12), Nstars)

'''    
# getting the DOmega in nHz
DOmega_LDR = DOmega_by_Omega_LDR * Omega

print('Omega in nHz:\n', Omega)
print('DOmega_by_Omega_gen:\n', DOmega_by_Omega_LDR)
print('Domega in nHz:\n', DOmega_LDR)  

print('Rotation periods in days:\n', Prot)
print('Omega in nHz:\n', Omega)
print('Delta Omega LDR in nHz:\n', Delta_Omega_LDR)
print('Delta Omega_LDR / Omega:\n', Delta_Omega_LDR/Omega)
'''
    
# plt.figure()
    
ax[1].loglog(Prot, DOmega_by_Omega_RDR_G2, 'k', label='G2')
ax[1].loglog(Prot, DOmega_by_Omega_RDR_K5, 'r', label='K5')
# plt.loglog(Prot, DOmega_by_Omega_RDR, 'x')
ax[1].set_xlabel('Rotation period (day)')
ax[1].set_label('RDR in nHz')
ax[1].set_title('RDR')
ax[1].legend()

plt.savefig('KR98_DR.pdf')
