import numpy as np
from tqdm import tqdm
from scipy import integrate
from scipy.interpolate import splev

from jax.experimental import sparse
import jax.numpy as jnp
from jax import jit

from jax_enseisro import globalvars as gvar_jax

import load_multiplets
import misc_functions_dpy as misc_fn
import wigner_map as wigmap
import prune_multiplets
import build_cenmults as build_cnm

_find_idx = wigmap.find_idx
jax_minus1pow_vec = misc_fn.jax_minus1pow_vec

# jitting the jax_gamma and jax_Omega functions
jax_Omega_ = jit(misc_fn.jax_Omega)
jax_gamma_ = jit(mics_fn.jax_gamma)


def get_s1_kernels(eig_idx, ell, GVARS, percomp_dict, s=1):
    '''Builds the integrated part of the fixed                                                
    part of pre-computation for the region below                                              
    rth.                                                                                      
    Parameters:                                                                               
    -----------                                                                               
    eig_idx : int                                                                             
              Index of the multiplet in the list of multiplets                                
              whose eigenfunctions are pre-loaded.                                            
                                                                                              
    ell     : int                                                                             
              Angular degree of the multiplet whose kernel integral                           
              we want to calculate.                                                           
                                                                                              
    s       : int                                                                             
              Angular degree of the perturbation                                              
              (differential rotation for now).                                                
    Returns:                                                                                  
    --------                                                                                  
    post_integral: float, ndarray                                                             
                   Array of shape containing the integrated values using                      
                   the fixed part of the profile below rth.                                   
    '''
    ls2fac = 2*ell*(ell+1) - s*(s+1)

    # slicing the required eigenfunctions
    U, V = precomp_dict.lm.U_arr[eig_idx], precomp_dict.lm.V_arr[eig_idx]

    # the factor in the integral dependent on eigenfunctions
    # shape (r,)
    eigfac = 2*U*V - U**2 - 0.5*(V**2)*ls2fac
    
    # total integrand
    integrand = -1. * eigfac / GVARS.r
    
    kernel_Omega1_out = integrate.trapz(integrand, GVARS.r)
    
    kernel_DeltaOmega1 = integrate.trapz(integrand[:precomp_dict.rcz_ind],
                                         GVARS.r[:precomp_dict.rcz_ind])
    
    # returns two scalars
    return kernel_Omega1_out, kernel_DeltaOmega1

def get_sgt1_kernels(eig_idx, ell, GVARS, percomp_dict, s):
    '''Builds the integrated part of the fixed                                                
    part of pre-computation for the region below                                              
    rth.                                                                                      
    Parameters:                                                                               
    -----------                                                                               
    eig_idx : int                                                                             
              Index of the multiplet in the list of multiplets                                
              whose eigenfunctions are pre-loaded.                                            
                                                                                              
    ell     : int                                                                             
              Angular degree of the multiplet whose kernel integral                           
              we want to calculate.                                                           
                                                                                              
    s       : int                                                                             
              Angular degree of the perturbation                                              
              (differential rotation for now).                                                
    Returns:                                                                                  
    --------                                                                                  
    post_integral: float, ndarray                                                             
                   Array of shape containing the integrated values using                      
                   the fixed part of the profile below rth.                                   
    '''
    ls2fac = 2*ell*(ell+1) - s*(s+1)

    # slicing the required eigenfunctions                                                    
    U, V = precomp_dict.lm.U_arr[eig_idx], precomp_dict.lm.V_arr[eig_idx]

    # the factor in the integral dependent on eigenfunctions                                 
    # shape (r,)                                                                             
    eigfac = 2*U*V - U**2 - 0.5*(V**2)*ls2fac

    # total integrand                                                                         
    integrand = -1. * eigfac / GVARS.r

    # sgt1 = s > 1
    kernel_DeltaOmega_sgt1 = integrate.trapz(integrand[precomp_dict.rcz_ind:],
                                         GVARS.r[precomp_dict.rcz_ind:])

    # returns two scalars                                                                     
    return kernel_DeltaOmega_sgt1

def build_hm_nonint_n_fxd_1cnm(s, CNM, precomp_dict):
    """Main function that does the multiplet-wise                                             
    precomputation of the non-c and the fixed part of the hypermatrix.                        
    In this case, the hypermatrix is effectively the diagonal of                              
    each cenmult which are appended one after another in a long                               
    column vector of length (2*ell+1).shape()                                                 
                                                                                            
    Paramters:                                                                                
    ----------                                                                                
    s : int                                                                                   
        The angualr degree for which the no-c and fixed part                                  
        needs to be precomputed.                                                              
                                                                                              
    Returns:                                                                                  
    --------                                                                                  
    non_c_diag_list   : float, ndarray in sparse form                                         
                        The pre-integrated part of the hypermatrix which                      
                        has the shape (nc x (2*ell+1).sum()).                                 
                                                                                              
    fixed_diag_sparse : float, ndarray in sparse form                                         
                        The pre-integrated part of the hypermatrix                            
                        which has the shape (2*ell+1).sum().
    """
    two_ellp1_sum_all = precomp_dict.num_cnm *\
                        (2 * precomp_dict.ellmax + 1)
    # the non-m part of the hypermatrix
    non_c_diag_arr = np.zeros((GVARS.nc, two_ellp1_sum_all))
    non_c_diag_list = []

    # the fixed hypermatrix (contribution below rth)
    fixed_diag_arr = np.zeros(two_ellp1_sum_all)

    start_cnm_ind = 0

    # filling in the non-m part using the masks
    for i in tqdm(range(precomp_dict.num_cnm), desc=f"Precomputing for s={s}"):
        # updating the start and end indices
        omega0 = CNM.omega_cnm[i]
        end_cnm_ind = start_cnm_ind + 2 * CNM.nl_cnm[i, 1] + 1

        # self coupling for isolated multiplets
        ell = CNM.nl_cnm[i, 1]

        wig1_idx, fac1 = _find_idx(ell, s, ell, 1)
        wigidx1ij = np.searchsorted(precomp_dict.wig_idx, wig1_idx)
        wigval1 = fac1 * precomp_dict.wig_list[wigidx1ij]

        m_arr = np.arange(-ell, ell+1)
        wig_idx_i, fac = _find_idx(ell, s, ell, m_arr)
        wigidx_for_s = np.searchsorted(precomp_dict.wig_idx, wig_idx_i)
        wigvalm = fac * precomp_dict.wig_list[wigidx_for_s]

        #-------------------------------------------------------
        # computing the ell1, ell2 dependent factors such as
        # gamma and Omega
        gamma_prod =  jax_gamma_(s) * jax_gamma_(ell)**2  
        Omega_prod = jax_Omega_(ell, 0)**2
        
        # also including 8 pi * omegaref
        omegaref = CNM.omega_cnm[i]
        ell1_ell2_fac = gamma_prod * Omega_prod *\
                        8 * np.pi * omegaref *\
                        (1 - jax_minus1pow_vec(s))

        # parameters for calculating the integrated part
        eig_idx = nl_idx_pruned.index(CNM.nl_cnm_idx[i])

        # shape (n_control_points,)
        # integrated_part = build_integrated_part(eig_idx1, eig_idx2, ell1, ell2, s)
        integrated_part = build_integrated_part(eig_idx, ell, s)
        #-------------------------------------------------------
        # integrating wsr_fixed for the fixed part
        fixed_integral = integrate_fixed_wsr(eig_idx, ell, s)

        wigvalm *= (jax_minus1pow_vec(m_arr) * ell1_ell2_fac)

        for c_ind in range(GVARS.nc):
            # non-ctrl points submat
            non_c_diag_arr[c_ind, start_cnm_ind: end_cnm_ind] =\
                                    integrated_part[c_ind] * wigvalm * wigval1

        # the fixed hypermatrix
        fixed_diag_arr[start_cnm_ind: end_cnm_ind] = fixed_integral * wigvalm * wigval1 

        # updating the start index
        start_cnm_ind = (i+1) * (2 * precomp_dict.ellmax + 1)

    # deleting wigvalm 
    del wigvalm, wigval1, fixed_integral, integrated_part

    # making it a list to allow easy c * hypermat later
    for c_ind in range(GVARS.nc):
        non_c_diag_arr_sparse = sparse.BCOO.fromdense(non_c_diag_arr[c_ind])
        non_c_diag_list.append(non_c_diag_arr_sparse)

    del non_c_diag_arr # deleting for ensuring no extra memory

    # sparsifying the fixed hypmat
    fixed_diag_sparse = sparse.BCOO.fromdense(fixed_diag_arr)
    del fixed_diag_arr             

    return non_c_diag_list, fixed_diag_sparse


def build_kernels_all_cenmults(CNM, precomp_dict):
    '''Precomputes all the arrays needed for the inversion.                                   
                                                                                              
    Returns:                                                                                  
    --------                                                                                  
    non_c_diag_cs : ndarray (sparse form), float                                              
                    Returns the sparse array of shape (s x c x (2*ell + 1).sum())             
                    containing the coefficients for s and each ctrl point.                    
                                                                                              
    fixed_diag :    ndarray (sparse form), float                                              
                    Returns the sparse array of shape (2*ell + 1).sum() containing            
                    the integrated fixed part of the flow profile below rth.                  
                                                                                              
    omega0_arr :    ndarray, float                                                            
                    Returns blocks of omega0 concatenated along one long                      
                    column of length (2*ell+1).sum() to be used later                         
                    when dividing by 2 * omega0.                                              
    '''
    # to store the cnm frequencies
    omega0_arr = np.ones(precomp_dict.num_cnm * (2 * precomp_dict.ellmax + 1))
    start_cnm_ind = 0
    for i, omega_cnm in enumerate(CNM.omega_cnm):
        # updating the start and end indices
        end_cnm_ind = start_cnm_ind + 2*CNM.nl_cnm[i,1] + 1 
        omega0_arr[start_cnm_ind:end_cnm_ind] *= CNM.omega_cnm[i]

        # updating the start index
        start_cnm_ind = (i+1) * (2 * precomp_dict.ellmax + 1)


    # stores the diags as a function of s and c. Shape (s x c) 
    non_c_diag_cs = []

    for s_ind, s in enumerate(GVARS.s_arr):
        # shape (dim_hyper x dim_hyper) but sparse form
        non_c_diag_s, fixed_diag_s =\
                    build_hm_nonint_n_fxd_1cnm(s, CNM, precompte_dict)
        
        # appending the different m part in the list
        non_c_diag_cs.append(non_c_diag_s)
        
        # adding up the different s for the fixed part
        if s_ind == 0:
            fixed_diag = fixed_diag_s
        else:
            fixed_diag += fixed_diag_s
    # non_c_diag_s = (s x 2ellp1_sum_all), fixed_diag = (2ellp1_sum_all,)
    return non_c_diag_cs, fixed_diag, omega0_arr
