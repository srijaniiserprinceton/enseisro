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
import sys

# ARGS = FN.create_argparser()
# GVAR = globalvars.globalVars(ARGS)

GVAR = globalvars.globalVars()

def plot_resolution_matrices(Nstars, smax, nmin, nmax, lmin, lmax):
    # building the multiplets from nmin, nmax, lmin, lmax
    mults = FN.build_mults(nmin, nmax, lmin, lmax)
    
    # getting the modes for which we want splitting values for one star
    # these would later be repeated to make multiple stars
    modes_single_star = make_modes.make_modes(mults)
    
    
    # building the star labels. Each star's label is repeated for all the modes
    # that is used from that star. Labelling starts from 0 so that we can use it
    # as array index later for getting rcz_ind and Omega_step_params, etc.
    star_label_arr, modes = FN.build_star_labels_and_all_modes(Nstars, modes_single_star)
    
    
    # creating the uncertainty of splitting vector                                                                                                                                     
    sigma_arr = 1 * np.ones(modes.shape[1])
    
    s_arr = np.arange(1,smax+1,2)
    lens = len(s_arr)

    # the rotation profile isn't necessary for getting the resolution matrices.
    # However, we compute some profile just to get the shape etc. correct so that 
    # the code smoothly returns the resolution matrices from the a_solve functions
    # which use the step_param_arr for getting certain lengths and parameters needed
    # for computing the A matrix and hence the resolution matrices.

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
    # converting it to Omega_{out} and \Delta \Omega
    step_param_arr = np.zeros_like(step_param_arr_1)
    # storing \Omega_{out}
    step_param_arr[:,:,0] = step_param_arr_1[:,:,1]
    # storing \Detla\Omega
    step_param_arr[:,:,1] = step_param_arr_1[:,:,0] - step_param_arr_1[:,:,1]
    
                                        
    # getting the data and model resolution matrices
    __, __, data_res_mat, model_res_mat = a_solver.use_numpy_inv_Omega_step_params(GVAR, star_label_arr, modes, sigma_arr,\
                                            smax, step_param_arr, rcz_ind_arr, use_diff_Omout=True, use_Delta=True, ret_res_mat=True)

    # plotting the resolution matrices
    fig, ax = plt.subplots(1, 2, figsize=(12,5.5))

    cmap = 'binary'

    # plotting the colormaps
    im_d = ax[0].pcolormesh(data_res_mat, cmap='seismic')
    im_m = ax[1].pcolormesh(model_res_mat, cmap='binary')

    fig.colorbar(im_d, ax=ax[0], shrink=0.85, aspect=40)
    fig.colorbar(im_m, ax=ax[1], shrink=0.85, aspect=40)

    ax[0].set_aspect('equal')
    ax[1].set_aspect('equal')
    
    ax[0].set_title('Data resolution matrix')
    ax[1].set_title('Model resolution matrix')
    
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    plt.tight_layout()

    plt.savefig('Resolution_matrices.pdf')

# some properties defining the number of modes
Nstars = 2
nmin, nmax = 2, 4
lmin, lmax = 2, 4

# properties defining the number of model parameters
smax = 3

plot_resolution_matrices(Nstars, smax, nmin, nmax, lmin, lmax)
