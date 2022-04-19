import numpy as np
import deepdish as dd

from jax_enseisro import globalvars as gvars_jax

# just to get access to the location of metadata                                   
GVARS_dummy = gvars_jax.GlobalVars()
ARGS = np.loadtxt(f'{GVARS_dummy.metadata}/.star_metadata.dat')

# creating the actual GVARS                                                     
GVARS = gvars_jax.GlobalVars(nStype=int(ARGS[0]),
                             nmin=int(ARGS[1]),
                             nmax=int(ARGS[2]),
                             lmin=int(ARGS[3]),
                             lmax=int(ARGS[4]),
                             smax=int(ARGS[5]),
                             rand_mults=int(ARGS[6]),
                             add_Noise=int(ARGS[7]),
                             use_Delta=int(ARGS[8]))

# loading the star mult array
star_mult_arr = dd.io.load(f'{GVARS.metadata}/star_mult_arr.h5')
# loading the array with number of stars in each Stype
num_startype_arr = np.load(f'{GVARS.metadata}/num_startype_arr.npy')

def num_rows_G():
    num_rows = 0
    for star_key in star_mult_arr.keys():
        # summing over the (2*ell+1) in each multiplet of each star within a star type
        num_rows += (2 * np.sum(star_mult_arr[f'{star_key}'][:,2]) +
                     len(star_mult_arr[f'{star_key}'][:,2]))
        
    return int(num_rows)

def num_cols_G():
    num_cols = 0
    
    # for each star there will be one column for the Omega_1_in
    # and then the rest of the columns will be for Delta_Omega_s
    # so total number of cols = Nstars + len(s_arr)

    num_cols = np.sum(num_startype_arr) + len(GVARS.s_arr)
    
    return int(num_cols)

def make_G(kernels):
    num_rows = num_rows_G()
    num_cols = num_cols_G()
    G = np.zeros((num_rows, num_cols))

    # number of Delta_Omega_s which are shared between the stars in ensemble
    len_s = len(GVARS.s_arr)

    # this controls which coloumn the Omega_1_in will be added to
    star_number = 0

    # looping over the multiplets to tile G matrix compactly
    start_ind_row_G = 0
    end_ind_row_G = 0
    
    # the star number in across Stypes. Always starts with star index 0.               
    star_num_this_Stype = 0

    print('G shape:', G.shape)

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

        # looping over the multiplets from the kernel matrix
        start_ind_row_K = 0
        end_ind_row_K = 0

        for mult_ind, ell in enumerate(ell_arr):
            end_ind_row_G = start_ind_row_G + 2*ell+1
            end_ind_row_K = start_ind_row_K + 2*ell+1
            
            print('G:', start_ind_row_G, end_ind_row_G)
            print('K:', start_ind_row_K, end_ind_row_K)

            # checking if we have moved onto a different star
            if(star_num_this_Stype == mult_arr[mult_ind,0]):
                pass
            # moving onto next star
            else:
                star_num_this_Stype = mult_arr[mult_ind,0]
                star_number += 1

            print(mult_arr[mult_ind], star_number)

            # assuming that only the first column is for Omega_1_in
            # which can differ across stars. Last cols are Delta_Omega_s which
            # are fixed for all the stars in the ensemble
            G[start_ind_row_G: end_ind_row_G, star_number] =\
                                kernel[start_ind_row_K: end_ind_row_K, 0]
            
            # filling in the Delta_Omega_s parts (which are the last cols of each row)
            G[start_ind_row_G: end_ind_row_G, -len_s:] =\
                                kernel[start_ind_row_K: end_ind_row_K, -len_s:]
            
            start_ind_row_G += 2*ell+1
            start_ind_row_K += twoellmaxp1        
    
    return G
