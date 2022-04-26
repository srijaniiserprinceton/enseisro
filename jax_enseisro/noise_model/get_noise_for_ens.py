import numpy as np
import sys

from jax_enseisro.noise_model import compute_libbrecht_noise as gen_libnoise

# {{{ def get_noise_for_ens():
def get_noise_for_ens(GVARS, star_mult_arr, mode_freq_arr, Teff_arr_stars,
                      g_arr_stars, numax_arr_stars, inc_angle_arr_stars):
    """Returns the sigma array of length Nmodes
    for an ensemble of stars. Can either return 
    all for replicas of the Sun or for different
    stars (in which case appropriate data files
    must be supplied).
    
    Paramters
    ---------
    star_mult_arr : list of numpy.ndarrays
                    List of arrays of shape (star_ind x n x ell) for 
                    each star type.
    
    mode_freq_arr : array_like, float
                    Array of mode frequencies in muHz for the modes used in inversion.
    
    Teff_arr_stars: array-like, float, shape (Nstars x 1)
                    Array of effective temperatures of the stars in the ensemble.
    
    g_arr_stars: array-like, float, shape (Nstars x 1)                                    
                    Array of surface gravities of the stars in the ensemble.
    
    numax_arr_stars: array-like, float, shape (Nstars x 1)
                    Array of freq of max power of the stars in the ensemble.
    """

    Nmodes = len(mode_freq_arr)
    
    # distributing the Teff arrays into the (Nmodes x 1) form from (Nstars x 1)
    Teff_arr = np.zeros(Nmodes)
    g_arr = np.zeros(Nmodes)
    numax_arr = np.zeros(Nmodes)
    inc_angle_arr = np.zeros(Nmodes)

    # filling in the arrays for these parameters for each star                              
    index_counter = 0
    
    # always stars with star_ind 0 for the first star
    star_ind = 0
    for star_key in star_mult_arr.keys():
        ell_arr_this_Stype = star_mult_arr[f'{star_ind}'][:,1]
        
        Nstars_this_Stype = np.unique(star_mult_arr[f'{star_ind}'][:,0])
        
        for star_in_Stype in Nstars_this_Stype:
            mask_this_star = star_mult_arr[f'{star_key}'][:,0] == star_in_Stype
            ell_arr_this_star = ell_arr_this_Stype[mask_this_star]
            
            twoellp1_sum_star = np.sum(2 * ell_arr_this_star + 1)
        
            Teff_arr[index_counter:index_counter+twoellp1_sum_star] =\
                                                    Teff_arr_stars[star_ind]
            g_arr[index_counter:index_counter+twoellp1_sum_star] =\
                                                    g_arr_stars[star_ind]
            numax_arr[index_counter:index_counter+twoellp1_sum_star] =\
                                                    numax_arr_stars[star_ind]
            inc_angle_arr[index_counter:index_counter+twoellp1_sum_star] =\
                                                    inc_angle_arr_stars[star_ind]
            
            # updating index counters
            index_counter += twoellp1_sum_star

    if(0 in Teff_arr):
        print('0 K Teff detected. Check Teff_arr filling.')
        sys.exit()
    if(0 in g_arr):
        print('0 surface gravity detected. Check g_arr filling.')
        sys.exit()
    if(0 in numax_arr):
        print('0 K numax detected. Check numax_arr filling.')
        sys.exit()
    

    # using the Libbrecht noise model to get \sigma(\delta \omega_nlm) in nHz               
    sigma_del_omega_nlm = gen_libnoise.compute_freq_uncertainties(GVARS, star_mult_arr,
                                                                  mode_freq_arr,
                                                                  Teff_arr, g_arr,
                                                                  numax_arr, inc_angle_arr)

    return sigma_del_omega_nlm
