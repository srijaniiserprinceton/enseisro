import numpy as np
from enseisro import globalvars
import enseisro.misc_functions as FN
from enseisro.synthetics import create_synth_modes as make_modes 
import matplotlib.pyplot as plt
from enseisro.functional_fitting import make_inversion_matrices as make_inv_mat
from enseisro.synthetics import create_synthetic_DR as create_synth_DR
from enseisro.synthetics import w_omega_functions as w_om_func
from enseisro import get_kernels as get_kerns 
from enseisro.functional_fitting import a_solver as a_solver
from enseisro import forward_functions_Omega as forfunc_Om
import matplotlib.pyplot as plt
import sys

ARGS = FN.create_argparser()
GVAR = globalvars.globalVars(ARGS)

#### testing the construction of the data matrix d ##################

# defining the multiplets
mults = np.array([[2,10], [2,2], [3,4], [4,5], [5,5]], dtype='int')
# creating the list of mults
nmin, nmax = 2, 15
lmin, lmax = 2, 15
mults = FN.build_mults(nmin, nmax, lmin, lmax)
#print(mults,mults.shape)
#sys.exit()

# getting the modes for which we want splitting values
modes = make_modes.make_modes(mults)

# creating the uncertainty of splitting vector                                                                                                                                          
sigma_arr = np.ones(modes.shape[1])

smax = 3
s_arr = np.arange(1,smax+1,2)
# extracting the solar DR profile in terms of wsr                                                                                                                                       
wsr = create_synth_DR.get_solar_DR(GVAR, smax=smax)
# converting this wsr to Omegasr                                                                                                                                                     
Omegasr = w_om_func.w_2_omega(GVAR, wsr)
# making it  in the (Nstars x s x r) shape
Omegasr = np.reshape(Omegasr, (1, len(s_arr), len(GVAR.r))) 


# finding the rcz index                           
rcz = 0.7
rcz_ind = np.argmin(np.abs(GVAR.r - rcz))
rcz_ind_arr = np.array([rcz_ind])   # contains only one star 

# calculating the step function equivalent of this Omegasr                                                                                                                              
step_param_arr = create_synth_DR.get_solar_stepfn_params(Omegasr, rcz_ind_arr)   # shape (Nstars x s x  Nparams)                                                                         
Omegasr_step = create_synth_DR.params_to_step(GVAR, step_param_arr, rcz_ind_arr)

# input synthetic values in nHz
print('Input synthetic values in nHz:', step_param_arr[0,:,:]*GVAR.OM*1e9)

# solving for a in A . a = d
a = a_solver.use_numpy_inv(GVAR, modes, sigma_arr, smax, use_synth=True, Omegasr = Omegasr_step[0])

# the inverted values in nHz
print('Inverted params in nHz:', a * GVAR.OM * 1e9)

