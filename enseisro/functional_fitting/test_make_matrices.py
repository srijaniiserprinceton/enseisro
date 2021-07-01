import numpy as np
from enseisro import globalvars
import enseisro.misc_functions as FN
from enseisro.synthetics import create_synth_modes as make_modes 
import matplotlib.pyplot as plt
from enseisro.functional_fitting import make_inversion_matrices as make_inv_mat
from enseisro.synthetics import create_synthetic_DR as create_synth_DR
from enseisro.synthetics import w_omega_functions as w_om_func
from enseisro import get_kernels as get_kerns 

ARGS = FN.create_argparser()
GVAR = globalvars.globalVars(ARGS)

#### testing the construction of the data matrix d ##################

# defining the multiplets
mults = np.array([[2,10], [2,12], [3,14]], dtype='int')

# getting the modes for which we want splitting values
modes = make_modes.make_modes(mults)

# extracting the solar DR profile in terms of wsr                                                                                                                                       
wsr = create_synth_DR.get_solar_DR(GVAR, smax=3)
# converting this wsr to Omegasr                                                                                                                                                     
Omegasr = w_om_func.w_2_omega(GVAR, wsr)


# getting the data matrix
d = make_inv_mat.make_d(GVAR, modes, use_synth=True, Omega_synth=Omegasr)

########## testing the construction of the A matrix ##################

# obtaining the kernel for a certain multiplet
K = get_kerns.compute_kernel(GVAR, np.array([[2,10]]), s_arr=np.array([1]))
print(K.shape)
