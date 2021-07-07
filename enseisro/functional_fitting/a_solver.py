# import jax.numpy as np
import numpy as np
import scipy as sp
from enseisro.functional_fitting import make_inversion_matrices as make_mat
from enseisro import forward_functions_Omega as forfunc_Om

def use_numpy_inv_Omega_function(GVAR, modes, sigma_arr, smax, use_synth=True, Omegasr=np.array([])):
    """This is to calculate a in the equation A . a = d, using the
    Numpy solver numpy.linalg.inv(). This uses Omega function of r"""
    
    A = make_mat.make_A(GVAR, modes, sigma_arr, smax=smax)   # shape (Nmodes x Nparams)
    d = make_mat.make_d_synth_from_function(GVAR, modes, sigma_arr, Omegasr)   # shape (Nmodes)

    
    AT = A.T

    ATA = AT @ A
    
    inv_ATA = sp.linalg.inv(ATA) 

    # computing solution
    a = inv_ATA @ (AT @ d)

    # computing the model covariance matrix.
    # C_M = (A^T A)^{-1} according to Sambridge
    # Lecture notes. Lecture 2, Slide 23.
    C_M = inv_ATA

    return a, C_M




def use_numpy_inv_Omega_step_params(GVAR, star_label_arr, modes, sigma_arr, smax, Omegas_step_params, rcz_ind_arr, use_Delta=True):
    """This is to calculate a in the equation A . a = d, using the
    Numpy solver numpy.linalg.inv(). This uses Omega step params"""
    
    Nstars = len(np.unique(star_label_arr))    # number of unique entries in the star label array
    lens = Omegas_step_params.shape[1]
    Nparams = (Omegas_step_params.shape[1] * Omegas_step_params.shape[2])  # since the combined dim is (s x Nparams) 
    # Nparams = (Nstars + lens)                  # this is the shape that we want to infer internal rotation separately 

    # getting the complete A matrix
    A = build_A_all_stars(GVAR, Nstars, modes, sigma_arr, Nparams, use_Delta, smax=smax)     # shape (Nmodes x Nparams)
    # getting the complete d vector
    d = build_d_all_stars(GVAR, Nstars, modes, sigma_arr, Omegas_step_params, \
                          rcz_ind_arr, use_Delta)       # shape (Nmodes)


    AT = A.T

    ATA = AT @ A
    
    inv_ATA = sp.linalg.inv(ATA) 

    # computing solution
    a = inv_ATA @ (AT @ d)

    # computing the model covariance matrix.
    # C_M = (A^T A)^{-1} according to Sambridge
    # Lecture notes. Lecture 2, Slide 23.
    C_M = inv_ATA

    return a, C_M


def build_A_all_stars(GVAR, Nstars, all_modes, sigma_arr, Nparams, use_Delta, smax=np.array([1])):
    """This function creates the A matrix accounting for multiple stars in the 
    ensemble."""
    Nmodes = all_modes.shape[1]
    modes_per_star = Nmodes // Nstars

    A = np.zeros((all_modes.shape[1], Nparams))

    if(use_Delta): make_A_function = make_mat.make_A_for_Delta_Omega
    else: make_A_function = make_mat.make_A
    
    
    # index for filling the matrices
    all_modes_label_start_ind, all_modes_label_end_ind = 0, modes_per_star 
    for i in range(Nstars):
        modes_star = all_modes[:,all_modes_label_start_ind: all_modes_label_end_ind]
        sigma_arr_star = sigma_arr[all_modes_label_start_ind: all_modes_label_end_ind]
       
        # filling in the A matrix
        A_star = make_A_function(GVAR, modes_star, sigma_arr_star, smax=smax)   # shape (Nmodes x Nparams)
        
        # filling in the correct rows
        A[all_modes_label_start_ind: all_modes_label_end_ind,:] = A_star
        
        # updating the start and end indices for filling the matrices
        all_modes_label_start_ind += modes_per_star 
        all_modes_label_end_ind += modes_per_star


    return A    # shape (Nmodes x Nparams)



def build_d_all_stars(GVAR, Nstars, all_modes, sigma_arr, Omegas_step_params, rcz_ind_arr, use_Delta):
    """This function creates the A matrix accounting for multiple stars in the 
    ensemble."""
    Nmodes = all_modes.shape[1]
    modes_per_star = Nmodes // Nstars

    d = np.zeros(all_modes.shape[1])
    
    # the function to evaluate the frequency splitting inside forward_functions_Omega
    # choosing it here avoid multiple executions of the if-else statements inside the loop
    if(use_Delta): compute_freq_splitting_fn = forfunc_Om.compute_splitting_from_step_params_for_Delta_Omega
    else: compute_freq_splitting_fn = forfunc_Om.compute_splitting_from_step_params


    # index for filling the matrices
    all_modes_label_start_ind, all_modes_label_end_ind = 0, modes_per_star 
    for i in range(Nstars):
        # getting the star-specific modes 
        modes_star = all_modes[:,all_modes_label_start_ind: all_modes_label_end_ind]
        # getting the star-specific uncertainties
        sigma_arr_star = sigma_arr[all_modes_label_start_ind: all_modes_label_end_ind]
        # getting the star-specific DR params 
        Omegas_step_params_star = Omegas_step_params[i]
        # getting the rcz_ind for the specfic-star
        rcz_ind_star = rcz_ind_arr[i]

        # building the data vector for a particular star
        d_star = make_mat.make_d_synth_from_step_params(GVAR, modes_star, sigma_arr_star,\
                         Omegas_step_params_star, rcz_ind_star, compute_freq_splitting_fn)   # shape (Nmodes x Nparams)
        
        # filling in the correct rows of the large data vector
        d[all_modes_label_start_ind: all_modes_label_end_ind] = d_star

        # updating the start and end indices for filling the matrices
        all_modes_label_start_ind += modes_per_star 
        all_modes_label_end_ind += modes_per_star

    return d                   # shape (Nmodes)
