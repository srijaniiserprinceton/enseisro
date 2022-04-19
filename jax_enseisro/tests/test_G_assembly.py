import numpy as np
import deepdish as dd

from jax_enseisro import globalvars as gvars_jax
from jax_enseisro.data_scripts import create_synthetic_mults
from jax_enseisro.inversion_scripts import make_kernels
from jax_enseisro.inversion_scripts import assemble_G_from_kernels as assemble_G

# just to get access to the location of metadata
GVARS_dummy = gvars_jax.GlobalVars()
ARGS = np.loadtxt(f'.star_metadata.dat')

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

'''Creating the multiplets used for each star. The order of storing
these multiplets is as follows:
{'Star1': [[Star1_n0, Star1_l0], [Star1_n1, Star1_l1], ....]
'Star2' : [[Star2_n0, Star2_l0], [Star2_n1, Star2_l1], ....]
.
.
.
'StarN' : [[StarN_n0, StarN_l0], [StarN_n1, StarN_l1], ....]}

This is in the form of a dictionary.
'''

# star-wise list of multiplets
star_mult_arr = dd.io.load(f'star_mult_arr.h5')

# get data vector
# data_vector = make_data_vector.get_d(star_mult_arr, GVARS)
kernels = make_kernels.make_kernels(star_mult_arr, GVARS)

# getting G matrix for forward problem
G = assemble_G.make_G(kernels)

# making model_params from Omega_step
Omega_step = np.load('Omega_step.npy')
Omega_1_in = Omega_step[0,0]
Delta_Omega_1 = Omega_step[0,-1] - Omega_step[0,0]
Delta_Omega_3 = Omega_step[1, -1]

model_params_K = np.array([Omega_1_in, Delta_Omega_1, Delta_Omega_3])
model_params_G = np.zeros(G.shape[1])
model_params_G[0:-2] = Omega_1_in
model_params_G[-2] = Delta_Omega_1
model_params_G[-1] = Delta_Omega_3

# checking if G ad kernels are consistent
freq_split_K = []
for star_key in star_mult_arr.keys():
    freq_split_K.append(model_params_K @ kernels[f'{star_key}'])
freq_split_K = np.asarray(freq_split_K).flatten()

freq_split_G = G @ model_params_G


def compress_freq_aplit_K(freq_split):
    '''Purges the extra zeros due to 2ellmaxp1 for each multiplet.
    Enables easy comparison with the freq_split_G.
    '''
    num_data = 0
    for star_key in star_mult_arr.keys():
        mult_arr = star_mult_arr[f'{star_key}']
        num_data += 2 * np.sum(mult_arr[:, 2]) + len(mult_arr[:, 2])
        
    freq_split_compressed = np.zeros(num_data)
    print(freq_split_compressed.shape)
    
    # defining the start and end indices for freq_split array
    start_ind_fs = 0
    end_ind_fs = 0
    
    # defining the start and end indices for freq_split_compressed array
    start_ind_fsc = 0 
    end_ind_fsc = 0
    
    for star_key in star_mult_arr.keys():
        ell_arr = star_mult_arr[f'{star_key}'][:, 2]
        
        twoellmaxp1 = 2 * np.max(ell_arr) + 1

        for mult_ind, ell in enumerate(ell_arr):
            end_ind_fs = start_ind_fs + 2*ell+1
            end_ind_fsc = start_ind_fsc + 2*ell+1
            
            print('fs:', start_ind_fs, end_ind_fs)
            print('fsc:', start_ind_fsc, end_ind_fsc)

            freq_split_compressed[start_ind_fsc: end_ind_fsc] =\
                                    freq_split[start_ind_fs: end_ind_fs]
            
            start_ind_fs += twoellmaxp1
            start_ind_fsc += 2*ell+1
    
    return freq_split_compressed    

freq_split_K2G = compress_freq_aplit_K(freq_split_K)
