# import jax.numpy as np
import numpy as np
from jax_enseisro.noise_model import misc_noise_functions as Noise_FN
from jax_enseisro.noise_model import get_Gamma as get_Gamma
from jax_enseisro.noise_model import convert_star_params as conv_star_params
from jax_enseisro.setup_scripts import misc_functions as misc_FN


# {{{ def compute_freq_uncertainties():
def compute_freq_uncertainties(GVARS, star_mult_arr, mode_freq_arr, Teff_arr,
                               g_arr, numax_arr, inc_angle_arr):
    """Returns the uncertainty in frequency for each mode
    according to Eqn.~(2.19) in Stahn's thesis.

    Parameters
    ----------
    modes : int, numpy.ndarray
            Array of modes of shape (3, Nmodes).                                            
                                                                          
    mode_freq_arr : float, array_like                                         
            Array of mode frequencies corresponding to ``modes`` in muHz. 
    """
   
    # values of the parameters taken from Table 2.1 in Stahn's thesis.                       
    A_arr = np.array([1.607, 0.542])       # amplitude array in ppm^2 \muHz^{-1}            
    A_err_arr = np.array([0.082, 0.030])   # error in amplitude array in ppm^2 \muHz^{-1}    
    tau_arr = np.array([1390.0, 455.0])    # time-scale array in seconds                     
    tau_arr_err = np.array([30, 10])       # error in time-scale array in seconds            
    
    # photon white noise in ppm^2 \muHz^{-1}                                                 
    # this is approx value Stahn's Fig 2.4 has. But he says that P_wn < 0.004. 
    P_wn = 0.00065    
    
    # getting the background noise profile B(\nu) in ppm^2/muHz
    N_nu = Noise_FN.make_N_nu(mode_freq_arr, tau_arr, A_arr, P_wn, return_harveys=False)
    c_bg = 1    # what is this value?? Not clear from Stahn's thesis
    B_nu_sun = c_bg * N_nu

    
    # getting the modes from the multiplets
    allstar_modes = None
    for key in star_mult_arr.keys():
        # (n, ell) for the Stype
        mult_arr = star_mult_arr[f'{key}'][:, 1:]
        
        if(int(key) == 0):
            allstar_modes = misc_FN.mults2modes(mult_arr)
        else:
            allstar_modes = np.append(allstar_modes,
                                      misc_FN.mults2modes(mult_arr), axis=1)

    n_arr, ell_arr, m_arr = allstar_modes[0,:], allstar_modes[1,:], allstar_modes[2,:]

    # getting linewidths in muHz
    Gamma_arr_sun = get_Gamma.get_Gamma(n_arr, ell_arr)
    
    print('Gamma_arr_sun: ', Gamma_arr_sun)
    print('Modes: ', allstar_modes)
    print('Mode freq array: ', mode_freq_arr)
    # computing the mode heights
    Hnlm_sun = Noise_FN.make_Hnlm(allstar_modes, mode_freq_arr, Gamma_arr_sun,
                                  inc_angle_arr)

    print('Hnlm_sun: ', Hnlm_sun)
    
    # uptil now we got the various params for the Sun. Now we convert
    # to other stars depending on scaling relations with T_eff, nu_max
    # and surface gravity
    
    rel_gravity_arr = g_arr / GVARS.g_sun
    rel_Teff_arr = Teff_arr / GVARS.Teff_sun
    rel_numax_arr = numax_arr / GVARS.numax_sun
    
    Hnlm = conv_star_params.convert_Hnlm(Hnlm_sun, rel_gravity_arr)
    Gamma_arr = conv_star_params.convert_Gamma(Gamma_arr_sun, rel_Teff_arr)
    B_nu = conv_star_params.convert_B_nu(B_nu_sun, rel_numax_arr)

    print('B_nu: ', B_nu)
    
    # \beta or the inverse signal-to-noise ratio
    beta = B_nu / Hnlm
    
    print('beta: ', beta)

    # f(\beta)
    one_plus_beta_sqrt = np.sqrt(1 + beta)
    beta_sqrt = np.sqrt(beta)
    fbeta = one_plus_beta_sqrt * (one_plus_beta_sqrt + beta_sqrt)**3

    # time T in micro sec. Considering 3 years
    T = GVARS.years_obs * (365 * 24 * 3600) * 1e-6   # the 1e-6 since we want 1/T in muHz

    # sigma^2 for mode nlm. In muHz^2 units.
    sigma_sq_omega_nlm = fbeta * Gamma_arr / (4 * np.pi * T)
    
    # sigma in nHz
    sigma_omega_nlm = np.sqrt(sigma_sq_omega_nlm) * 1e3

    # creating the sigma for frequency splitting (\delta \omega)
    sigma_del_omega_nlm = sigma_omega_nlm / np.abs(m_arr)

    print('sigma_del_omega_nlm: ', sigma_del_omega_nlm)

    return sigma_del_omega_nlm

# }}} def compute_freq_uncertainties()
    
