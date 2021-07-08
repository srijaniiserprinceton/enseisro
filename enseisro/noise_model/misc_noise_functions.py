# import jax.numpy as np
import numpy as np

# this file contains miscellaneous functions to construct the Noise model.
# the nomenclature is adopted from Thorsten Stahn's PhD thesis.

# {{{ def make_N_nu():
def make_N_nu(nu_arr, tau_arr, A_arr, P_wn, return_harveys=False):
    """This function returns $N(\nu)$ from a given frequency array $\nu$.

    Parameters
    ----------
    nu_arr : numpy.ndarray    
        Frequency array in micro Hertz.
    tau_arr : numpy.ndarray
        Time-scale array in seconds.
    A_arr : numpy.ndarray
        Amplitude array in ppm^2 muHz^{-1}
    P_wn : scalar
        Photon white noise in ppm^2 muHz^{-1}
    return_harvets : boolean
        True or False depending on whether or not we want to 
        return the Harvey models separately. This is mainly used for 
        plotting them separately when tallying with Stahn's Figure 2.4.
    """
    ncomponents = 2      # since there are two components
    
    # converting the nu_arr to Hz since that is what we will use
    # in the equation, the denominator is 1 + (\tau_i * \nu)**4
    # so, \tau and \nu dimnensions must be consistent
    nu_arr = 1e-6 * nu_arr    # converting muHz to Hz

    # array to store the functions N(\nu)
    N_nu_arr = np.zeros_like(nu_arr)
    harvey_models = np.zeros((2, len(nu_arr)))

    for i in range(ncomponents):
        harvey_models[i] = A_arr[i]/(1 + (tau_arr[i] * nu_arr)**4)
        N_nu_arr += harvey_models[i]

    # adding the photon white noise
    N_nu_arr += P_wn

    if(return_harveys): return N_nu_arr, harvey_models
    else: return N_nu_arr

# }}} def make_N_nu()
