# import jax.numpy as np
import numpy as np
from enseisro.noise_model import compute_libbrecht_noise as gen_libnoise
from enseisro.functional_fitting import a_solver as a_solver
from enseisro import misc_functions as FN
from enseisro.synthetics import create_synth_modes as make_modes
from enseisro import globalvars
from enseisro.synthetics import create_synthetic_DR as create_synth_DR
from enseisro.synthetics import w_omega_functions as w_om_func
from enseisro import forward_functions_Omega as forfunc_Om
from enseisro.functional_fitting import make_inversion_matrices as make_inv_mat
GVAR = globalvars.globalVars()

def test_noise_model():
    # number of stars
    Nstars = 1
    
    # generating the modes
    # defining the multiplets                                                                                                                                                          
    nmin, nmax = 16, 18
    lmin, lmax = 2, 2
    mults = FN.build_mults(nmin, nmax, lmin, lmax)

    # modes for a single star
    modes_single_star = make_modes.make_modes(mults)
    # getting all modes across all stars
    star_label_arr, modes = FN.build_star_labels_and_all_modes(Nstars, modes_single_star)


    # making the rotation profile
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

    # using Delta Omega and Omega_out instead of Omega_in and Omega_out
    use_Delta = True

    # noise free data (omega_nlm) generation
    sigma_arr = np.ones(modes.shape[1])


    # getting the frequencies corresponding to the modes
    # getting the complete mode frequencies which is the d vector                                                                                                                                                 
    delta_mode_freq_arr = a_solver.build_d_all_stars(GVAR, Nstars, modes, sigma_arr, step_param_arr, \
                                               rcz_ind_arr, use_Delta)       # shape (Nmodes)
    
    # getting the absolute frequencies from delta omega_nlm
    mult_idx = FN.nl_idx_vec(GVAR, mults)                                                                                                                                             
    omeganl = GVAR.omega_list[mult_idx]
    
    

    # to store the omega_nlm for all modes
    omeganl_arr = np.zeros(modes.shape[1])
    
    # looping over the multiplets and storing
    current_index  = 0  # keeping a counter on index position
    for mult_ind, mult in enumerate(mults):
        n_inst, ell_inst = mult[0], mult[1]
        inst_mult_marr = make_inv_mat.get_inst_mult_marr(n_inst, ell_inst, modes)
        Nmodes_in_mult = len(inst_mult_marr)
        omeganl_arr[current_index:current_index + Nmodes_in_mult] = omeganl[mult_ind]
        current_index += Nmodes_in_mult

    # total freq = omeganl + delta omega_nlm
    mode_freq_arr = omeganl_arr + delta_mode_freq_arr
        
    # we need to pass the mode_freq_arr in muHz
    mode_freq_arr *= (GVAR.OM * 1e6)
    
    # getting the Teff, surface gravity and numax for other stars
    Teff_stars = np.zeros(Nstars) + GVAR.Teff_sun 
    g_stars = np.zeros(Nstars) + GVAR.g_sun
    numax_stars = np.zeros(Nstars) + GVAR.numax_sun

    Teff_arr = np.zeros(modes.shape[1])
    g_arr = np.zeros(modes.shape[1])
    numax_arr = np.zeros(modes.shape[1])

    # filling in the arrays for these parameters for each star
    index_counter = 0
    Nmodes_per_star = modes_single_star.shape[1]
    for i in range(Nstars):
        Teff_arr[index_counter:index_counter+Nmodes_per_star] = Teff_stars[i]
        g_arr[index_counter:index_counter+Nmodes_per_star] = g_stars[i]
        numax_arr[index_counter:index_counter+Nmodes_per_star] = numax_stars[i]

    # using the Libbrecht noise model to get \sigma(\delta \omega_nlm) in nHz
    sigma_del_omega_nlm = gen_libnoise.compute_freq_uncertainties(modes, mode_freq_arr,\
                                                              Teff_arr, g_arr, numax_arr)

    # printing the modes
    print('Modes:\n',modes)
    # in nHz
    print('Sigma_omega_nlm in nHz:\n', sigma_del_omega_nlm)

test_noise_model()
