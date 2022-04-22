import os
import numpy as np

from jax_enseisro import globalvars as gvars_jax

current_dir = os.path.dirname(os.path.realpath(__file__))
jax_enseisro_dir = os.path.dirname(current_dir)

metadata_path = f'{jax_enseisro_dir}/inversion_metadata'

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

# the data dictionary
data_dict = {}
data_dict['data'] = np.load(f'{GVARS.synthdata}/synthetic_data.npy')
data_dict['sigma_d'] = np.load(f'{GVARS.synthdata}/sigma_d.npy') * 1e-3

# the model dictionary
model_dict = {}
model_dict['G'] = np.load(f'{GVARS.metadata_path}/G.npy')
model_dict['model_init'] = np.load(f'{GVARS.metadata_path}/model_params_true.npy') * 0.0
model_dict['model_ref'] = np.load(f'{GVARS.metadata_path}/model_params_true.npy')

# the regularization dictionary
reg_dict = {}
reg_dict['mu'] = 1.

# the loop dictionary
loop_dict = {}
loop_dict['loss_threshold'] = 1e-12
loop_dict['maxiter'] = 20

# the miscellaneous field dictionary
misc_dict = {}
misc_dict['hessinv'] = None
