# import jax.numpy as np
import numpy as np
from scipy.special import lpmv
from math import factorial

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
    return_harveys : boolean, optional
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


# {{{ def mode_visibility():
def mode_visibility(abs_m_arr, ell_arr, inc_angle_arr):
    """This computes the visibility of individual m-components
    depending on the inclination angle i, of the rotation axis
    with respect to the line of sight.

    Parameters
    ----------
    abs_m_arr : int, array_like
        Array containing |m|.
    ell_arr : int, array_like
        Array containing \ell.
    inc_angle_arr : float, array_like
        Array contiaining the inclination angle of the rotation axis 
        of the star with respect to the line of sight. Array has the 
        same length as the length of the abs_m_arr or ell_arr.
    """

    # vectorizing the math.factorial function
    fac_vec = np.vectorize(factorial)
    prefac = fac_vec(ell_arr - abs_m_arr)/fac_vec(ell_arr + abs_m_arr) 

    eps_ell_m = prefac * lpmv(abs_m_arr, ell_arr, inc_angle_arr)**2

    # should have the same length as the ell_arr or abs_m_arr
    return eps_ell_m
# }}} def mode_visibility()

# {{{ def envelope_function():
def envelope_function(mode_freq, cen_freq, sigma_1, sigma_2):
    """Calculating the envelope function as given by Eqn.~(2.6) 
    in Stahn's thesis.
    
    Parameters
    ----------
    mode_freq : float, array_like
        The mode frequencies \nu_{n,\ell,m}
    cen_freq : float
        The frequency of the center of the amplitude envelope.
    sigma_1 : float
        Width of the envelope below cen_freq.
    sigma_2 : float
        Width of the envelope above cen_freq.
    """
    # creating a sigma array depending on relative value of
    # mode_freq and cen_freq
    sigma_arr = np.zeros_like(mode_freq)

    ind_where_modefreq_leq_cenfreq = mode_freq < cen_freq
    sigma_arr[ind_where_modefreq_leq_cenfreq] = sigma_1
    sigma_arr[~ind_where_modefreq_leq_cenfreq] = sigma_2
    
    # has the same shape as the mode_freq
    F = (1 + ((mode_freq - cen_freq)/sigma_arr)**2)**(-2)

    return F
