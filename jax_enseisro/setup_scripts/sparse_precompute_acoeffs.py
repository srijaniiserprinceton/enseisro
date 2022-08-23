import os
import numpy as np
from tqdm import tqdm
from scipy import integrate
from scipy.interpolate import splev
import deepdish as dd

from jax.experimental import sparse
import jax.numpy as jnp
from jax import jit

from jax_enseisro import globalvars as gvar_jax

from jax_enseisro.setup_scripts import load_multiplets
from jax_enseisro.setup_scripts import misc_functions as misc_fn
from jax_enseisro.setup_scripts import wigner_map as wigmap

_find_idx = wigmap.find_idx
jax_minus1pow_vec = misc_fn.jax_minus1pow_vec

# jitting the jax_gamma and jax_Omega functions
jax_Omega_ = jit(misc_fn.jax_Omega)
jax_gamma_ = jit(misc_fn.jax_gamma)

current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)

metadata_path = f'{package_dir}/inversion_metadata'

RL_Poly = dd.io.load(f'{metadata_path}/RL_poly.h5')

def get_s1_kernels(eig_idx, ell, precomp_dict, s=1):
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
    # does not contain 1 / r since we use Omega_s instead of w_s
    integrand = -1. * eigfac
    
    kernel_Omega1_out = integrate.trapz(integrand, precomp_dict.r)
    
    kernel_DeltaOmega1 = integrate.trapz(integrand[precomp_dict.rcz_idx:],
                                         precomp_dict.r[precomp_dict.rcz_idx:])
    
    # returns two scalars
    return np.array([kernel_Omega1_out, kernel_DeltaOmega1])

def get_sgt1_kernels(eig_idx, ell, precomp_dict, s=3):
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
    # does not contain 1/ r since we use Omega_s instead of w_s
    integrand = -1. * eigfac

    # sgt1 = s > 1
    kernel_DeltaOmega_sgt1 = integrate.trapz(integrand[precomp_dict.rcz_idx:],
                                             precomp_dict.r[precomp_dict.rcz_idx:])

    # returns a scalar                                                                     
    return np.array([kernel_DeltaOmega_sgt1])

def build_kernel_each_s(s, CNM, precomp_dict):
    """Main function that does the multiplet-wise                                             
    precomputation of the non-c and the fixed part of the hypermatrix.                        
    In this case, the hypermatrix is effectively the diagonal of                              
    each cenmult which are appended one after another in a long                               
    column vector of length (2*ell+1).shape()                                                 
                                                                                            
    Paramters:                                                                                
    ----------                                                                                
    s : int                                                                                   
        The angular degree for which the no-c and fixed part                                  
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

    if(s == 1):
        # fits for both \Omega1_out and \Delta\Omega1
        kernel_arr = np.zeros((2, precomp_dict.num_cnm))
        kernel_fn = get_s1_kernels
    else:
        # fits for only \Omega_s_out
        kernel_arr = np.zeros((1, precomp_dict.num_cnm))
        kernel_fn = get_sgt1_kernels
        

    # filling in the non-m part using the masks
    for i in tqdm(range(precomp_dict.num_cnm),
                  desc=f"[Type {precomp_dict.Stype}] Precomputing for s={s}"):
        # updating the start and end indices
        omega0 = CNM.omega_cnm[i]

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
        # does not contain the jax_gamma_(s) factor since we use Omega_s instead of w_s
        gamma_prod = jax_gamma_(ell)**2  
        Omega_prod = jax_Omega_(ell,0)**2
        
        # also including 8 pi * omegaref
        # omegaref = CNM.omega_cnm[i]

        ell1_ell2_fac = gamma_prod * Omega_prod *\
                        4 * np.pi *\
                        (1 - jax_minus1pow_vec(s))

        # parameters for calculating the integrated part
        eig_idx = precomp_dict.nl_idx_pruned.index(CNM.nl_cnm_idx[i])

        #-------------------------------------------------------
        # integrating and returning appropriate kernel depending on s
        kernel = kernel_fn(eig_idx, ell, precomp_dict, s=s)

        wigvalm *= (jax_minus1pow_vec(m_arr) * ell1_ell2_fac)

        # this is where the conversion would happen from 
        # wigvalm to wigval_acoeff
        # wigvalm = wigval_acoeff * Psl(m)
        # \sum_m (wigvalm * Psl(m)) / (\sum_m Psl(m) * Psl(m)) = wigval_acoeff

        Psl_m = RL_Poly[f'{ell}'][f'{s}']
        wigval_acoeff = wigvalm @ Psl_m / (Psl_m @ Psl_m)
        
        # filling the kernel array
        for j in range(kernel_arr.shape[0]):
            kernel_arr[j,i] = kernel[j] * wigval_acoeff * wigval1 


    # deleting wigvalm 
    del wigval_acoeff, wigval1, kernel

    # shape = (2 x num_cnm) for s = 1 and (1 - num_cnm for s > 1
    return kernel_arr


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


    # stores the kernels in (num_s + 1 x num_cnm) form
    # the +1 comes from the fact that only s=1 has two components.
    kernel_arr_all_s = np.zeros((len(precomp_dict.s_arr)+1,
                                 precomp_dict.num_cnm))

    for s_ind, s in enumerate(precomp_dict.s_arr):
        # shape (2 x num_cnm) for s = 1 and (1 x num_cnm) for s > 1
        kernel_arr = build_kernel_each_s(s, CNM, precomp_dict)
        
        if(s == 1):
            kernel_arr_all_s[:2] = kernel_arr
        else:
            kernel_arr_all_s[s_ind+1] = kernel_arr
        
    return kernel_arr_all_s
