# import jax.numpy as np
import numpy as np
from scipy.special import lpmv
from math import factorial

# this file contains miscellaneous functions to construct the Noise model.
# the nomenclature is adopted from Thorsten Stahn's PhD thesis.

# {{{ def make_N_nu():
def make_N_nu(nu_arr, tau_arr, A_arr, P_wn, return_harveys=False):
    """This function returns $N(\nu)$ from a given frequency array ``nu``.

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
        Array containing absolute value of the azimuthal degree.
    ell_arr : int, array_like
        Array containing angular degree.
    inc_angle_arr : float, array_like
        Array contiaining the inclination angle of the rotation axis 
        of the star with respect to the line of sight. Array has the 
        same length as the length of the abs_m_arr or ell_arr.
    """

    # vectorizing the math.factorial function
    fac_vec = np.vectorize(factorial)
    prefac = fac_vec(ell_arr - abs_m_arr)/fac_vec(ell_arr + abs_m_arr) 

    eps_ell_m = prefac * lpmv(abs_m_arr, ell_arr, np.cos(inc_angle_arr))**2

    # should have the same length as the ell_arr or abs_m_arr
    return eps_ell_m
# }}} def mode_visibility()

# {{{ def envelope_function():
def envelope_function(mode_freq_arr, cen_freq, sigma_1, sigma_2):
    """Calculating the envelope function as given by Eqn.~(2.6) 
    in Stahn's thesis.
    
    Parameters
    ----------
    mode_freq_arr : float, array_like
        The mode frequencies nu_{n,\ell,m} in muHz.
    cen_freq : float
        The frequency of the center of the amplitude envelope in muHz.
    sigma_1 : float
        Width of the envelope below cen_freq in muHz.
    sigma_2 : float
        Width of the envelope above cen_freq in muHz.
    """
    # creating a sigma array depending on relative value of
    # mode_freq and cen_freq
    sigma_arr = np.zeros_like(mode_freq_arr)

    ind_where_modefreq_leq_cenfreq = mode_freq_arr < cen_freq
    sigma_arr[ind_where_modefreq_leq_cenfreq] = sigma_1
    sigma_arr[~ind_where_modefreq_leq_cenfreq] = sigma_2
    
    # has the same shape as the mode_freq
    F = (1 + ((mode_freq_arr - cen_freq)/sigma_arr)**2)**(-2)

    return F


# {{{ def make_Hnlm():                                                                                                                                                                
def make_Hnlm(modes, mode_freq_arr, Gamma_arr, inc_angle_arr=None):
    """Returns H_nlm computed according to Eqn.~(2.14)                                                                                           
    in Stahn's thesis.

    Parameters
    ----------
    modes : int, numpy.ndarray
        Array of modes of shape (3, Nmodes).
    mode_freq_arr : float, array_like
        Array of mode frequencies corresponding to ```modes``` in muHz.
    Gamma_arr : float, array_like
        Array of linewidths of the ```modes``` in muHz.
    inc_angle_arr : float, optional
        Array of inclination angles of the stellar rotation axis with respect
        to the line of sight. This may vary across stars. Default is set to 90 degrees.
    """

    # the inclination array. Elements of this array may vary across stars                                                                                                      
    # as a simple case, we may choose it to be 0 degrees                                                                                                                               
    if(inc_angle_arr == None):
        inc_angle_arr = np.zeros_like(mode_freq_arr) + np.pi/2.0

    # the maximum of the amplitude envelope for modes                                                                                                                                   
    # with a particular angular degree ell                                                                                                                                
    A_ell = np.array([4.64, 6.08, 3.69])    # values in ppm taken from Table 2.3 in Stahn's thesis                                                                                      
    A_ell_err = np.array([0.13, 0.15, 0.11])

    # the widths on either side of cen_freq. Values are in muHz                                                                                                                         
    sigma_1, sigma_2 = 564, 678
    sigma_1_err, sigma_2_err = 36, 28

    # central mode of amplitude envelope in muHz                                                                                                                                        
    cen_freq = 3067
    cen_freq_err = 26

    # getting the array for n, ell and abs_m                                                                                                                                            
    n_arr = modes[0,:]
    ell_arr = modes[1,:]
    abs_m_arr = np.abs(modes[2,:])


    # the mode visibility. Same length as the mode_freq_arr or the number of modes                                                                                                      
    # E_lm_i = mode_visibility(abs_m_arr, ell_arr, inc_angle_arr)
    E_lm_i = np.ones_like(mode_freq_arr)
    

    # the envelope function. Same length as the mode_freq_arr or the number of modes                                                                                                    
    F_nlm = envelope_function(mode_freq_arr, cen_freq, sigma_1, sigma_2)

    # making the A_ell_arr for the modes. We essentially store the A_ell values in                                                                                                     
    # corresponding modes                                                                                                                                                               
    A_ell_arr = np.zeros_like(mode_freq_arr)
    A_ell_arr[ell_arr == 0] = A_ell[0]
    A_ell_arr[ell_arr == 1] = A_ell[1]
    A_ell_arr[ell_arr == 2] = A_ell[2]

    # Hnlm array                                                                                                                                                                        
    Hnlm = np.zeros_like(mode_freq_arr)

    # computing the Hnlm                                                                                                                                                                
    Hnlm = (A_ell_arr**2/ (np.pi * Gamma_arr)) * E_lm_i * F_nlm

    # Same length as the mode_freq_arr or the number of modes                                                                                                             
    return Hnlm

# }}} def make_Hnlm() 
