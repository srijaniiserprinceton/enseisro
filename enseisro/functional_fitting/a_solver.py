# import jax.numpy as np
import numpy as np
import scipy as sp
from enseisro.functional_fitting import make_inversion_matrices as make_mat
from enseisro import forward_functions_Omega as forfunc_Om
from enseisro import misc_functions as FN
import copy

def use_numpy_inv_Omega_function(GVAR, modes, sigma_arr, smax, use_synth=True, Omegasr=np.array([]),
                                 ret_res_mat=False, add_noise=False):
    """This is to calculate a in the equation A . a = d, using the
    Numpy solver numpy.linalg.inv(). This uses Omega function of r"""
    
    A = make_mat.make_A(GVAR, modes, sigma_arr, smax)   # shape (Nmodes x Nparams)
    d = make_mat.make_d_synth_from_function(GVAR, modes, sigma_arr, Omegasr)   # shape (Nmodes)

    
    # adding noise is necessary
    if(add_noise):
        d += FN.gen_freq_splitting_noise(sigma_arr)

    print(d * GVAR.OM * 1e9)
    
    AT = A.T

    ATA = AT @ A
    
    inv_ATA = sp.linalg.inv(ATA) 

    # computing the generalized inverse G^{-g}
    gen_inv = inv_ATA @ AT

    # computing solution
    a = gen_inv @ d

    # computing the model covariance matrix.
    # C_M = (A^T A)^{-1} according to Sambridge
    # Lecture notes. Lecture 2, Slide 23.
    C_M = inv_ATA

        
    # is resolution matrices are also requested
    if(ret_res_mat):
        # data resolution matrix 
        data_res_mat = A @ gen_inv 

        # model resolution matrix
        model_res_mat = gen_inv @ A

        return a, C_M, data_res_mat, model_res_mat

    else: return a, C_M



def use_numpy_inv_Omega_step_params(GVAR, modes, sigma_arr, smax, Omegas_step_params, rcz_ind_arr,\
                                    use_diff_Omout=True, use_Delta=True, ret_res_mat=False, add_noise=False):
    """This is to calculate a in the equation A . a = d, using the
    Numpy solver numpy.linalg.inv(). This uses Omega step params"""
    
    Nstars = Omegas_step_params.shape[0]    # Omegas_step_params has the shape (Nstars x s x Nparams_per_star)
    lens = Omegas_step_params.shape[1]

    
    # this is the shape if we want to infer internal rotation separately for each star but \Delta Omega is shared
    # Params = {Om^(1)_{out,1}, Om^(3)_{out,1},...,Om^(1)_{out,N}, Om^(3)_{out,N}, \Delta Om^(1), \Delta Om^(3)}
    if(use_diff_Omout): 
        Nparams = (Nstars + 1)*lens
        build_A_function = build_A_all_stars_diff_Omout
    # this is the shape if we want to infer internal rotation also as a shared parameter
    else: 
        Nparams = (lens * Omegas_step_params.shape[2])  # since the combined dim is (s x Nparams_each_s) 
        build_A_function = build_A_all_stars_same_Omout
        
    # getting the complete A matrix
    A = build_A_function(GVAR, Nstars, modes, sigma_arr, Nparams, use_Delta, smax)     # shape (Nmodes x Nparams)
    
    AT = A.T

    ATA = AT @ A
    
    inv_ATA = sp.linalg.inv(ATA) 

    # generalized inverse matrix. Usually written as G^{-g}
    gen_inv = inv_ATA @ AT
    
    # getting the complete d vector using the forward problem
    d = build_d_all_stars(GVAR, Nstars, modes, sigma_arr, Omegas_step_params, \
                          rcz_ind_arr, use_Delta)       # shape (Nmodes)

    # adding noise is necessary
    if(add_noise):
        d += FN.gen_freq_splitting_noise(sigma_arr)
    

    # computing solution
    a = gen_inv @ d

    # computing the model covariance matrix.
    # C_M = (A^T A)^{-1} according to Sambridge
    # Lecture notes. Lecture 2, Slide 23.
    C_M = inv_ATA
    
    
    # is resolution matrices are also requested
    if(ret_res_mat):
        # data resolution matrix 
        data_res_mat = A @ gen_inv 

        # model resolution matrix
        model_res_mat = gen_inv @ A

        return a, C_M, data_res_mat, model_res_mat

    else: return a, C_M


def build_A_all_stars_same_Omout(GVAR, Nstars, all_modes, sigma_arr,\
                                 Nparams, use_Delta, smax):
    """This function creates the A matrix accounting for multiple stars in the 
    ensemble when they have the same properties, i.e., \Omega_{out} and \Delta \Omega"""
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
        A_star = make_A_function(GVAR, modes_star, sigma_arr_star, smax)   # shape (Nmodes x Nparams)
        
        # filling in the correct rows
        A[all_modes_label_start_ind: all_modes_label_end_ind,:] = A_star
        
        # updating the start and end indices for filling the matrices
        all_modes_label_start_ind += modes_per_star 
        all_modes_label_end_ind += modes_per_star


    return A    # shape (Nmodes x Nparams)




def build_A_all_stars_diff_Omout(GVAR, Nstars, all_modes, sigma_arr,\
                                 Nparams, use_Delta, smax):
    """This function creates the A matrix accounting for multiple stars in the 
    ensemble when they share the same \Delta\Omega but different \Omega_{out}"""
    Nmodes = all_modes.shape[1]
    modes_per_star = Nmodes // Nstars

    lens = (smax - 1)//2 + 1
    
    # shape (Nmodes x (lens * (Nstars+1)))
    A = np.zeros((all_modes.shape[1], Nparams))

    if(use_Delta): make_A_function = make_mat.make_A_for_Delta_Omega
    else: make_A_function = make_mat.make_A
    
    
    # index for filling the matrices
    all_modes_label_start_ind, all_modes_label_end_ind = 0, modes_per_star 
    for i in range(Nstars):
        modes_star = all_modes[:,all_modes_label_start_ind: all_modes_label_end_ind]
        sigma_arr_star = sigma_arr[all_modes_label_start_ind: all_modes_label_end_ind]
       
        # filling in the A matrix
        A_star = make_A_function(GVAR, modes_star, sigma_arr_star, smax)   # shape (Nmodes x Nparams)
        
        # filling in the Omout part of the kernel
        star_label = i
       
        # slicing out the Omout parts carefully. They are arranged as (Omout_1, Delta Omega_1, Omout_3, Delta Omega_3)
        A[all_modes_label_start_ind: all_modes_label_end_ind, lens*star_label:lens*star_label+lens] = A_star[:,0:-1:2]        

        # filling in the \Delta Omega part of the kernel
        A[all_modes_label_start_ind: all_modes_label_end_ind, -lens:] = A_star[:,1::2]

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
