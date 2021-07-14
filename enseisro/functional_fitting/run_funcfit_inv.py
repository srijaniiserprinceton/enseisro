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
from enseisro.synthetics import create_rot_prof_KR98 as create_rot_prof
import sys


GVAR = globalvars.globalVars()

def run_ens_inv(Prot, Nstars, nmin, nmax, lmin, lmax, smax, rcz_arr, p, use_Delta=True, add_noise=False):
    ################### CREATING MULTIPLETS, MODES AND STAR LABELS ##########################
    
    # defining the multiplets
    # mults = np.array([[2,10], [2,2], [3,4], [4,5], [5,5]], dtype='int')
    # creating the list of mults
    mults_single_star = FN.build_mults_single_star(nmin, nmax, lmin, lmax)
    
    # getting the modes for which we want splitting values for one star
    # these would later be repeated to make multiple stars
    modes_single_star = make_modes.make_modes(mults_single_star)
    
    # building the star labels. Each star's label is repeated for all the modes
    # that is used from that star. Labelling starts from 0 so that we can use it
    # as array index later for getting rcz_ind and Omega_step_params, etc.
    star_multlabel_arr, mults, star_modelabel_arr, modes = FN.build_star_labels_and_mults_modes(Nstars, mults_single_star, modes_single_star)
    
    ########################### CREATING ROTATION PROFILES ###################################
    
    smax = smax
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
    
    
    # finding the rcz index for all the stars
    rcz_ind_arr = np.zeros(Nstars, dtype='int')
    for i in range(Nstars):                           
        rcz_ind = np.argmin(np.abs(GVAR.r - rcz_arr[i]))
        rcz_ind_arr[i] = rcz_ind              
        
    # calculating the step function equivalent of this Omegasr. (\Omega_{in} + \Omega_{out))                                                                                                                          
    step_param_arr_in_out = create_synth_DR.get_solar_stepfn_params(Omegasr, rcz_ind_arr)   # shape (Nstars x s x  Nparams)                                                                         
    # adding constant random terms to each star's rotation profiles keeping \Delta \Omega_s constant 
    step_param_arr_in_out = create_synth_DR.randomize_DR_step_params(step_param_arr_in_out, profile_type='in-out', p=p)
    
    '''
    # converting it to Omega_{out} and \Delta \Omega
    step_param_arr = np.zeros_like(step_param_arr_in_out)
    # storing \Omega_{out}
    step_param_arr[:,:,0] = step_param_arr_in_out[:,:,1]
    # storing \Detla\Omega
    step_param_arr[:,:,1] = step_param_arr_in_out[:,:,0] - step_param_arr_in_out[:,:,1]
    '''
    # getting the step_param_arr in nHz
    step_param_arr = create_rot_prof.make_step_param_arr(Nstars, Prot, spectral_type='G2')
   
    # adding constant random terms to each star's rotation profiles keeping \Delta \Omega_s constant 
    step_param_arr = create_synth_DR.randomize_DR_step_params(step_param_arr, profile_type='Delta', p=p)

    # converting to non-dimensional units
    step_param_arr = step_param_arr / (GVAR.OM * 1e9)

    ##################### CREATING SYNTHETIC NOISE FOR FREQUENCY SPLITTINGS ##################
    
    if(add_noise):
        # getting the frequencies corresponding to the modes without noise
        sigma_arr_no_noise = np.ones(modes.shape[1])                                                                                                     
        # getting the complete mode frequencies which is the d vector
        delta_omega_nlm_arr = a_solver.build_d_all_stars(GVAR, Nstars, modes, sigma_arr_no_noise, step_param_arr, \
                                                         rcz_ind_arr, use_Delta)       # shape (Nmodes)                                                                                   
        
        # getting the absolute frequencies from delta omega_nlm
        omega_nlm_arr = FN.get_omega_nlm_from_delta_omega_nlm(GVAR, delta_omega_nlm_arr, Nstars, mults, modes,\
                                                              star_multlabel_arr, star_modelabel_arr)
        
        # creating the synthetic sigma for frequency splitting measurements
        sigma_arr_nhz = get_noise.get_noise_for_ens(GVAR, Nstars, modes, omega_nlm_arr, Nmodes_single_star=modes_single_star.shape[1])
        
        # converting to non-dimensional units
        sigma_arr = sigma_arr_nhz / (GVAR.OM * 1e9)
    
    else:
        sigma_arr = np.ones(modes.shape[1]) / (GVAR.OM * 1e9)
    

    ################################################# INVERSIONS ##########################################################
    
    # solving for a in A . a = d
    # Also calculating the uncertainty in the inverted model paramters
    # computing the model covariance matrix C_M
    # from Sambridge Lecture notes (Lecture 2, Slide 23)
    # C_M = (G^T C_d^{-1} G)^{-1}. In our case G^T C_d^{1/2} = A^T
    # C_d^{1/2} = \sigma
    
    a_Delta, C_M_Delta = a_solver.use_numpy_inv_Omega_step_params(GVAR, modes, sigma_arr, smax, step_param_arr, rcz_ind_arr,\
                                                use_diff_Omout=True, use_Delta=use_Delta, ret_res_mat=False, add_noise=add_noise)
    
    # converting to nHz
    a_Delta = a_Delta * GVAR.OM * 1e9
    C_M_Delta = C_M_Delta * (GVAR.OM * 1e9)**2
    
    ##################################################### PRINTING OUTPUT ##################################################
    
    # creating the various arrays to be used for printing
    synthetic_out = step_param_arr.flatten()[::2] * GVAR.OM * 1e9
    synthetic_delta = step_param_arr.flatten()[1:2*lens:2] * GVAR.OM * 1e9
    inverted_out = a_Delta[:-lens] 
    inverted_delta = a_Delta[-lens:]
    
    # printing the outputs
    line_breaks = '\n\n\n'
    print(line_breaks)
    print('Number of stars: ', Nstars)
    print('Number of modes per star: ', modes_single_star.shape[1])
    print('Synthetic noise added: ', add_noise)
    print(line_breaks)
    # printing the formatted output table
    print_func.format_terminal_output_ens_star(Nstars, synthetic_out, synthetic_delta, inverted_out, inverted_delta, smax)
    print(line_breaks)


    # the diagonal model covariance matrix in nHz
    error_arr =  np.sqrt(np.diag(C_M_Delta))
    # print('Model covariance matrix in nHz:\n', error_arr)
    # print(line_breaks)

    
    # average Omega in ensemble
    if(smax == 1):
        Omega_avg = np.mean(a_Delta[:-1])
        err_Omega_avg = np.mean(error_arr[:-1])
        DOmega = a_Delta[-1]
        err_DOmega = error_arr[-1]
        
        # Nstar multiple to see how many is necessary
        Nstar_multiple = np.array([1, 10, 50, 100, 500, 1000])

        for i in range(len(Nstar_multiple)):
            fac = Nstar_multiple[i]
            print('For %i stars:'%(fac * Nstars))
            print('Avg ensemble Omega in nHz: ', Omega_avg, ' +/- ', err_Omega_avg/np.sqrt(fac))
            print('Avg ensemble DOmega in nHz: ', DOmega, ' +/- ', err_DOmega/np.sqrt(fac))
            print('\n')
            
        return Omega_avg, DOmega, err_Omega_avg, err_DOmega

    

if __name__ == '__main__':
    # defining the parameters necessary
    # Number of stars
    Nstars = 10
    
    # Observed mode info
    nmin, nmax = 16, 23
    lmin, lmax = 2, 2  

    # Max angular degree for Omega_s
    smax = 1

    # Base of the convection zone for each star
    rcz_arr = np.zeros(Nstars) + 0.7

    # the rotation period of the stars in days                                                                                                                                               
    Prot_low, Prot_high = 25, 27
    # Prot = np.linspace(Prot_low, Prot_high, Nstars)
    Prot = np.array([26])  

    # the max. perentage randomization of rotation profiles
    p = 10 
    
    # whether to use the noise model to add synthetic noise 
    # to the frequency splitting data
    add_noise = True

    # carrying out the inversion and printing the results
    run_ens_inv(Prot, Nstars, nmin, nmax, lmin, lmax, smax, rcz_arr, p, add_noise=add_noise)
