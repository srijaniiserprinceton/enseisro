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
# mults = np.array([[2,10], [2,2], [3,4], [4,5], [5,5]], dtype='int')
# creating the list of mults
nmin, nmax = 2, 4
lmin, lmax = 2, 4
mults = FN.build_mults(nmin, nmax, lmin, lmax)
#print(mults,mults.shape)
#sys.exit()

# getting the modes for which we want splitting values
modes = make_modes.make_modes(mults)

# creating the uncertainty of splitting vector                                                                                                                                          
sigma_arr = 0.01 * np.ones(modes.shape[1])

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


# solving for a in A . a = d

# Also calculating the uncertainty in the inverted model paramters
# computing the model covariance matrix C_M
# from Sambridge Lecture notes (Lecture 2, Slide 23)
# C_M = (G^T C_d^{-1} G)^{-1}. In our case G^T C_d^{1/2} = A^T
# C_d^{1/2} = \sigma

a, C_M = a_solver.use_numpy_inv(GVAR, modes, sigma_arr, smax, use_synth=True, Omegasr = Omegasr_step[0])

# printing the outputs
line_breaks = '\n\n\n'
print(line_breaks)
print('Number of modes used: ', modes.shape[1])
print('Number of inversion parameters: ', len(a))
print(line_breaks)
# printing the formatted output table
FN.format_terminal_output(step_param_arr[0,:,:].flatten()*GVAR.OM*1e9, a.flatten()*GVAR.OM*1e9, np.sqrt(np.diag(C_M)))
print(line_breaks)
# the model covariance matrix
print('Model covariance matrix in nHz^2:\n', C_M)
print(line_breaks)
