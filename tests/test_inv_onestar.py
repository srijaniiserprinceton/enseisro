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

# ARGS = FN.create_argparser()
# GVAR = globalvars.globalVars(ARGS)

GVAR = globalvars.globalVars()

def test_inv_onestar():
    """This function is used to test if a single star inversion
    works fine. This could alternatively be used for multiple stars
    which have the same \Omega_{out} and \Delta \Omega."""
    # defining the number of stars in the ensemble
    Nstars = 1
    
    # defining the multiplets
    # mults = np.array([[2,10], [2,2], [3,4], [4,5], [5,5]], dtype='int')
    # creating the list of mults
    nmin, nmax = 2, 2
    lmin, lmax = 2, 3
    mults = FN.build_mults(nmin, nmax, lmin, lmax)
    
    # getting the modes for which we want splitting values for one star
    # these would later be repeated to make multiple stars
    modes_single_star = make_modes.make_modes(mults)
    
    
    # building the star labels. Each star's label is repeated for all the modes
    # that is used from that star. Labelling starts from 0 so that we can use it
    # as array index later for getting rcz_ind and Omega_step_params, etc.
    star_label_arr, modes = FN.build_star_labels_and_all_modes(Nstars, modes_single_star)
    
    
    # creating the uncertainty of splitting vector                                                                                                                                          
    sigma_arr = 0.01 * np.ones(modes.shape[1])
    
    smax = 3
    s_arr = np.arange(1,smax+1,2)
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
    
    Omegasr_step = create_synth_DR.params_to_step(GVAR, step_param_arr, rcz_ind_arr) # shape (Nstars x s x r)
    
    # solving for a in A . a = d
    
    # Also calculating the uncertainty in the inverted model paramters
    # computing the model covariance matrix C_M
    # from Sambridge Lecture notes (Lecture 2, Slide 23)
    # C_M = (G^T C_d^{-1} G)^{-1}. In our case G^T C_d^{1/2} = A^T
    # C_d^{1/2} = \sigma
    
    a_Delta, C_M_Detla = a_solver.use_numpy_inv_Omega_step_params(GVAR, star_label_arr, modes, sigma_arr,\
                                smax, step_param_arr, rcz_ind_arr, use_diff_Omout=False, use_Delta=True)
    a, C_M = a_solver.use_numpy_inv_Omega_step_params(GVAR, star_label_arr, modes, sigma_arr,\
                            smax, step_param_arr_1, rcz_ind_arr, use_diff_Omout=False, use_Delta=False)
    
    
    np.testing.assert_array_almost_equal(a_Delta.flatten(), step_param_arr[0].flatten())
    np.testing.assert_array_almost_equal(a.flatten(), step_param_arr_1[0].flatten())
    
    '''
    print('Synthetic:\n',step_param_arr_1[0].flatten() * GVAR.OM * 1e9)
    print('Inverted:\n', a.flatten() * GVAR.OM * 1e9)
    print('Synthetic with Delta:\n',step_param_arr[0].flatten() * GVAR.OM * 1e9)
    print('Inverted with Delta:\n',a_Delta.flatten() * GVAR.OM * 1e9) # , a_1 * GVAR.OM * 1e9)
    '''

# test_inv_onestar()
