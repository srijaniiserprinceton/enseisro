import numpy as np
import matplotlib.pyplot as plt
from enseisro.synthetics import create_rot_prof_KR98 as create_rot_prof

# creating the range of Prot we want to use
Prot = np.logspace(np.log10(1), np.log10(30), 10)

# getting the Delta Omega and Omega in nHz
Delta_Omega, Omega = create_rot_prof.get_DeltaOmega_from_Prot(Prot, 'G2')

print('Rotation periods in days:\n', Prot)
print('Omega in nHz:\n', Omega)
print('Delta Omega in nHz:\n', Delta_Omega)
print('Delta Omega / Omega:\n', Delta_Omega/Omega)
