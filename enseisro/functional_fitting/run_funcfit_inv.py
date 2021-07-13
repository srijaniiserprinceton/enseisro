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
from enseisro import printing_functions as print_func
from enseisro.noise_model import get_noise_for_ens as get_noise
import sys

# ARGS = FN.create_argparser()
# GVAR = globalvars.globalVars(ARGS)

GVAR = globalvars.globalVars()

# defining the number of stars in the ensemble
Nstars = 10

# defining the multiplets
# mults = np.array([[2,10], [2,2], [3,4], [4,5], [5,5]], dtype='int')
# creating the list of mults
nmin, nmax = 16, 18
lmin, lmax = 2, 2
mults_single_star = FN.build_mults_single_star(nmin, nmax, lmin, lmax)

# getting the modes for which we want splitting values for one star
# these would later be repeated to make multiple stars
modes_single_star = make_modes.make_modes(mults_single_star)


# building the star labels. Each star's label is repeated for all the modes
# that is used from that star. Labelling starts from 0 so that we can use it
# as array index later for getting rcz_ind and Omega_step_params, etc.
star_multlabel_arr, mults, star_modelabel_arr, modes = FN.build_star_labels_and_mults_modes(Nstars, mults_single_star, modes_single_star)


smax = 1
s_arr = np.arange(1,smax+1,2)
lens = len(s_arr)
# extracting the solar DR profile in terms of wsr                                                                                                                                       
wsr = create_synth_DR.get_solar_DR(GVAR, smax=smax)
# converting this wsr to Omegasr                                                                                                                                                     
Omegasr = w_om_func.w_2_omega(GVAR, wsr)
# making it  in the (1 x s x r) shape
Omegasr = np.reshape(Omegasr, (1, len(s_arr), len(GVAR.r))) 
# making multiple identical copies for different stars. Shaoe (Nstars x s x r) 
Omegasr = np.tile(Omegasr,(Nstars,1,1))

# creating rcz for all the stars. Does not need to be the same value
rcz_arr = np.zeros(Nstars) + 0.7

# finding the rcz index for all the stars
rcz_ind_arr = np.zeros(Nstars, dtype='int')
for i in range(Nstars):                           
    rcz_ind = np.argmin(np.abs(GVAR.r - rcz_arr[i]))
    rcz_ind_arr[i] = rcz_ind              

# calculating the step function equivalent of this Omegasr. (\Omega_{in} + \Omega_{out))                                                                                                                          
step_param_arr_1 = create_synth_DR.get_solar_stepfn_params(Omegasr, rcz_ind_arr)   # shape (Nstars x s x  Nparams)                                                                         
# adding constant random terms to each star's rotation profiles keeping \Delta \Omega_s constant 
step_param_arr_1 = create_synth_DR.randomize_DR_step_params(step_param_arr_1, p=10)


# converting it to Omega_{out} and \Delta \Omega
step_param_arr = np.zeros_like(step_param_arr_1)
# storing \Omega_{out}
step_param_arr[:,:,0] = step_param_arr_1[:,:,1]
# storing \Detla\Omega
step_param_arr[:,:,1] = step_param_arr_1[:,:,0] - step_param_arr_1[:,:,1]


# getting the frequencies corresponding to the modes without noise
sigma_arr_no_noise = np.ones(modes.shape[1])                                                                                                     
# getting the complete mode frequencies which is the d vector
use_Delta = True                                                                                                                           
delta_mode_freq_arr = a_solver.build_d_all_stars(GVAR, Nstars, modes, sigma_arr_no_noise, step_param_arr, \
                                                 rcz_ind_arr, use_Delta)       # shape (Nmodes)                                                                                   

# getting the absolute frequencies from delta omega_nlm                                                                                                                       
mult_idx = FN.nl_idx_vec(GVAR, mults)
omeganl = GVAR.omega_list[mult_idx]


# to store the omega_nl for all modes                                                                                                                                     
omeganl_arr = np.zeros(modes.shape[1])

# looping over the multiplets and storing                                                                                                                                              
current_index  = 0  # keeping a counter on index position                                                                                                                                                      

for i in range(Nstars):
    mult_star_ind = (star_multlabel_arr == i)
    mult_star = mults[mult_star_ind]
    for mult_ind, mult in enumerate(mult_star):
        n_inst, ell_inst = mult[0], mult[1]
        modes_current_star = modes[:,star_modelabel_arr==i]
        inst_mult_marr = FN.get_inst_mult_marr(n_inst, ell_inst, modes_current_star)
        Nmodes_in_mult = len(inst_mult_marr)
        omeganl_arr[current_index:current_index + Nmodes_in_mult] = omeganl[mult_ind]
        current_index += Nmodes_in_mult


# total freq = omeganl + delta omega_nlm                                                                                                                                        
mode_freq_arr = omeganl_arr + delta_mode_freq_arr

# we need to pass the mode_freq_arr in muHz                                                                                                                                
mode_freq_arr *= (GVAR.OM * 1e6)



sigma_arr = get_noise.get_noise_for_ens(GVAR, Nstars, modes, mode_freq_arr, Nmodes_single_star=modes_single_star.shape[1])

print('Sigma array:\n', sigma_arr)

sigma_arr = np.ones_like(sigma_arr)

# sys.exit()

# solving for a in A . a = d

# Also calculating the uncertainty in the inverted model paramters
# computing the model covariance matrix C_M
# from Sambridge Lecture notes (Lecture 2, Slide 23)
# C_M = (G^T C_d^{-1} G)^{-1}. In our case G^T C_d^{1/2} = A^T
# C_d^{1/2} = \sigma

a_Delta, C_M_Delta = a_solver.use_numpy_inv_Omega_step_params(GVAR, modes, sigma_arr,\
                     smax, step_param_arr, rcz_ind_arr, use_diff_Omout=True, use_Delta=True, ret_res_mat=False)

# creating the various arrays to be used for printing
synthetic_out = step_param_arr.flatten()[::2] * GVAR.OM * 1e9
synthetic_delta = step_param_arr.flatten()[1:2*lens:2] * GVAR.OM * 1e9
inverted_out = a_Delta[:-lens] * GVAR.OM * 1e9
inverted_delta = a_Delta[-lens:] * GVAR.OM * 1e9

# printing the outputs
line_breaks = '\n\n\n'
print(line_breaks)
print('Number of stars: ', Nstars)
print('Number of modes per star: ', modes_single_star.shape[1])
print(line_breaks)
# printing the formatted output table
print_func.format_terminal_output_ens_star(Nstars, synthetic_out, synthetic_delta, inverted_out, inverted_delta, smax)
print(line_breaks)


# the model covariance matrix
print('Model covariance matrix in nHz:\n', np.sqrt(np.diag(C_M_Delta)))
print(line_breaks)
