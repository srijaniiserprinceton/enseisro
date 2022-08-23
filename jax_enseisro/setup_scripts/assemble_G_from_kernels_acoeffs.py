import numpy as np
import deepdish as dd

from jax_enseisro import globalvars as gvars_jax

def num_rows_G(star_mult_arr, is_acoeffs_kernel):
    num_rows = 0
    
    if(is_acoeffs_kernel):
        for star_key in star_mult_arr.keys():
            # summing over the (2*ell+1) in each multiplet of each star within a star type
            num_rows += len(star_mult_arr[f'{star_key}'][:,2])
    else:
        for star_key in star_mult_arr.keys():
            # summing over the (2*ell+1) in each multiplet of each star within a star type 
            num_rows += (2 * np.sum(star_mult_arr[f'{star_key}'][:,2]) +
                         len(star_mult_arr[f'{star_key}'][:,2]))
                        
    return int(num_rows)

def num_cols_G(GVARS):
    num_cols = 0
    
    # for each star there will be one column for the Omega_1_in
    # and then the rest of the columns will be for Delta_Omega_s
    # so total number of cols = Nstars + len(s_arr)

    num_cols = np.sum(GVARS.num_startype_arr) + len(GVARS.s_arr)
    
    return int(num_cols)

def make_G(kernels, GVARS, star_mult_arr):
    num_rows = num_rows_G(star_mult_arr, GVARS.is_acoeffs_kernel)
    num_cols = num_cols_G(GVARS)
    G = np.zeros((num_rows, num_cols))

    # number of Delta_Omega_s which are shared between the stars in ensemble
    len_s = len(GVARS.s_arr)

    # this controls which coloumn the Omega_1_in will be added to
    star_number = 0

    # looping over the multiplets to tile G matrix compactly
    ind_row_G = 0
    
    # the star number in across Stypes. Always starts with star index 0.               
    star_num_this_Stype = 0

    for star_key in kernels.keys():        
        # the kernel for this Stype (contains multiple stars)
        kernel = kernels[f'{star_key}']
        
        # making the kernel of shape (modes x model_params) from (model_params x modes)
        kernel = np.transpose(kernel)

        ellmax = np.max(star_mult_arr[f'{star_key}'][:,2])
        twoellmaxp1 = 2 * ellmax + 1

        # the array of multiplets (star_no, n, ell) for the star type
        mult_arr = star_mult_arr[f'{star_key}']
        # the array of angular degrees ell for the star type
        ell_arr = mult_arr[:, 2]

        
        for mult_ind, ell in enumerate(ell_arr):
            # checking if we have moved onto a different star
            if(star_num_this_Stype == mult_arr[mult_ind,0]):
                pass
            # moving onto next star
            else:
                star_num_this_Stype = mult_arr[mult_ind,0]
                star_number += 1

            # assuming that only the first column is for Omega_1_in
            # which can differ across stars. Last cols are Delta_Omega_s which
            # are fixed for all the stars in the ensemble
            G[ind_row_G, star_number] = kernel[mult_ind, 0]
            
            # filling in the Delta_Omega_s parts (which are the last cols of each row)
            G[ind_row_G, -len_s:] = kernel[mult_ind, -len_s:]
            
            ind_row_G += 1
    
    return G
