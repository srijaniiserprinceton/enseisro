# import jax.numpy as np
import numpy as np
from enseisro.noise_model import compute_libbrecht_noise as gen_libnoise

# {{{ def get_noise_for_ens():
def get_noise_for_ens(GVAR, Nstars, modes, mode_freq_arr, Nmodes_single_star=None):
    """Returns the sigma array of length Nmodes
    for an ensemble of stars. Can either return 
    all for replicas of the Sun or for different
    stars (in which case appropriate data files
    must be supplied).
    
    Paramters
    ---------
    Nstars : scalar, int
        Number of stars in the ensemble.
    modes : numpy.ndarray, int
        Array of modes in the shape (n x ell x m).
    mode_freq_arr : array_like, float
        Array of mode frequencies in muHz for the modes used in inversion.
    Nmodes_single_star : int, optional
        Number of modes per star.
    """

    Nmodes = len(mode_freq_arr)

    # getting the Teff, surface gravity and numax for other stars                                                                                                                        
    Teff_stars = np.zeros(Nstars) + GVAR.Teff_sun
    g_stars = np.zeros(Nstars) + GVAR.g_sun
    numax_stars = np.zeros(Nstars) + GVAR.numax_sun

    Teff_arr = np.zeros(Nmodes)
    g_arr = np.zeros(Nmodes)
    numax_arr = np.zeros(Nmodes)

    # filling in the arrays for these parameters for each star                                                                                                                           
    index_counter = 0
    for i in range(Nstars):
        Teff_arr[index_counter:index_counter+Nmodes_single_star] = Teff_stars[i]
        g_arr[index_counter:index_counter+Nmodes_single_star] = g_stars[i]
        numax_arr[index_counter:index_counter+Nmodes_single_star] = numax_stars[i]

        # updating index counters
        index_counter += Nmodes_single_star


    # using the Libbrecht noise model to get \sigma(\delta \omega_nlm) in nHz                                                                                                            
    sigma_del_omega_nlm = gen_libnoise.compute_freq_uncertainties(GVAR, modes, mode_freq_arr,\
                                                              Teff_arr, g_arr, numax_arr)

    return sigma_del_omega_nlm
