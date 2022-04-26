import numpy as np
import deepdish as dd

from jax_enseisro import globalvars as gvars_jax
from jax_enseisro.noise_model import get_noise_for_ens as get_noise
from data_scripts import create_synthetic_mults
# from data_scripts import make_data_vector
from setup_scripts import make_kernels
from setup_scripts import assemble_G_from_kernels as assemble_G
from setup_scripts import make_model_params as make_model_params

metadata_path = './inversion_metadata'

# just to get access to the location of metadata
ARGS = np.loadtxt(f'{metadata_path}/.star_metadata.dat')

# creating the actual GVARS
GVARS = gvars_jax.GlobalVars(nStype=int(ARGS[0]),
                             nmin=int(ARGS[1]),
                             nmax=int(ARGS[2]),
                             lmin=int(ARGS[3]),
                             lmax=int(ARGS[4]),
                             smax=int(ARGS[5]),
                             rand_mults=int(ARGS[6]),
                             add_Noise=int(ARGS[7]),
                             use_Delta=int(ARGS[8]),
                             metadata_path=metadata_path)

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
star_mult_arr = dd.io.load(f'{metadata_path}/star_mult_arr.h5')

# get data vector
# data_vector = make_data_vector.get_d(star_mult_arr, GVARS)
kernels = make_kernels.make_kernels(star_mult_arr, GVARS)

# getting G matrix for forward problem
G = assemble_G.make_G(kernels, GVARS, star_mult_arr)

# making synthetic model params
num_model_params = G.shape[1]

# making model_params from Omega_step stored in parent "tests/" directory                  
Omega_step = np.load(f'Omega_step.npy')
model_params_G = make_model_params.make_model_params(Omega_step,
                                                     num_model_params, GVARS)

# making the synthetic data
synth_data = G @ model_params_G

# reading the Teff, surface gravity and numax arrays
Teff_arr_stars = np.load(f'{GVARS.synthdata}/Teff_arr_stars.npy')
g_arr_stars = np.load(f'{GVARS.synthdata}/g_arr_stars.npy')
numax_arr_stars = np.load(f'{GVARS.synthdata}/numax_arr_stars.npy')

# making the synthetic noise
# synth_noise = np.ones_like(synth_data)
synth_noise = get_noise.get_noise_for_ens(GVARS, star_mult_arr, synth_data,
                                          Teff_arr_stars, g_arr_stars, numax_arr_stars)

'''
# to add synthetic noise to synthetic data
if(GVARS.add_Noise == 1): synth_data = 
'''

#------------------SAVING THE PRECOMPUTED ARRAYS------------------#
# saving the precomputed G matrix
np.save(f'{GVARS.metadata_path}/G.npy', G)

# saving the true model params
np.save(f'{GVARS.metadata_path}/model_params_true.npy', model_params_G)

# saving the synthetic data
np.save(f'{GVARS.synthdata}/synthetic_data.npy', synth_data)

# saving the synthetic noise
np.save(f'{GVARS.synthdata}/sigma_d.npy', synth_noise)
