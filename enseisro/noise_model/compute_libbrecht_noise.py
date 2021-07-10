# import jax.numpy as np
import numpy as np
from enseisro.noise_model import misc_noise_functions as Noise_FN
from enseisro.noise_model import get_Gamma as get_Gamma

# {{{ def compute_freq_uncertainties():
def compute_freq_uncertainties(modes, mode_freq_arr):
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
    P_wn = 0.00065    # this is approx value Stahn's Fig 2.4 has. But he says that P_wn < 0.004. 
    
    # getting the background noise profile B(\nu) in ppm^2/muHz
    N_nu = Noise_FN.make_N_nu(mode_freq_arr, tau_arr, A_arr, P_wn, return_harveys=False)
    c_bg = 1    # what is this value?? Not clear from Stahn's thesis
    B_nu = c_bg * N_nu

    
    # getting the linewidths for the modes
    n_arr, ell_arr, m_arr = modes[0,:], modes[1,:], modes[2,:]

    # getting linewidths in muHz
    Gamma_arr = get_Gamma.get_Gamma(n_arr, ell_arr)
    
    
    # computing the mode heights
    Hnlm = Noise_FN.make_Hnlm(modes, mode_freq_arr, Gamma_arr)

    # \beta or the inverse signal-to-noise ratio
    beta = B_nu / Hnlm

    # f(\beta)
    one_plus_beta_sqrt = np.sqrt(1 + beta)
    beta_sqrt = np.sqrt(beta)
    fbeta = one_plus_beta_sqrt * (one_plus_beta_sqrt + beta_sqrt)**3

    # time T in micro sec. Considering 3 years
    T = 3 * (365 * 24 * 3600) * 1e-6   # the 1e-6 since we want 1/T in muHz

    

    # sigma^2 for mode nlm. In muHz^2 units.
    sigma_sq_omega_nlm = fbeta * Gamma_arr / (4 * np.pi * T)
    
    # sigma in nHz
    sigma_omega_nlm = np.sqrt(sigma_sq_omega_nlm) * 1e3

    # creating the sigma for frequency splitting (\delta \omega)
    sigma_del_omega_nlm = sigma_omega_nlm / np.abs(m_arr)

    return sigma_del_omega_nlm

# }}} def compute_freq_uncertainties()
    
