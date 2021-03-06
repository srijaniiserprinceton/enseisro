import os
import numpy as np
import deepdish as dd

from jax_enseisro import globalvars as gvars_jax
from jax_enseisro.data_scripts import create_synthetic_mults

# getting the absolute path of the current directory                
current_dir = os.path.dirname(os.path.realpath(__file__))

'''This file is used to specify the metadata. FOr the 
synthetic problem, this file also saves the information of
Star Type (SType), number of starts in the start type (nStype),
and the set of multiplets available in each star.

nStype : The number of types of stars in the ensemble. The background
         model in each start type will be varying -> different rcz.

num_startype_arr : The number of stars under each star type. The sum total of 
            nstar_arr would give the total number of stars in the ensemble.

rand_mults : Whether to introduce minor changes across the radial order of
             multiplets used in each star. 
'''

#-------------------------- defining metadata -----------------------------#
nStype = 4

num_startype_arr = np.array([2, 3, 4, 5], dtype='int')

# rcz_startype_arr = np.array([0.68, 0.71, 0.7, 0.69])
rcz_startype_arr = np.array([0.7, 0.7, 0.7, 0.7])

nmin, nmax = 16, 24
lmin, lmax = 1, 3

rand_mults = 0

# Max angular degree for Omega_s
smax = 3

# whether to use the noise model to add synthetic noise                                  
# to the frequency splitting data
add_Noise = 0

# whether to use (Omega_in, Omega_out) or (Omega_out, Delta_Omega)
use_Delta = 1

metadata_path = f'{current_dir}'

#------writing out the metadata to be used in other files-------------------#

with open(f"{metadata_path}/.star_metadata.dat", "w") as f:
    f.write(f"{nStype}" + "\n" +
            f"{nmin}"+ "\n" +
            f"{nmax}" + "\n" +
            f"{lmin}" + "\n" +
            f"{lmax}" + "\n" + 
            f"{smax}" + "\n" +
            f"{rand_mults}" + "\n" +
            f"{add_Noise}" + "\n" +
            f"{use_Delta}")

# storing the number of stars in each type
np.save(f'{metadata_path}/num_startype_arr.npy', num_startype_arr)

# storing the array of rcz by star type
np.save(f'{metadata_path}/rcz_startype_arr.npy', rcz_startype_arr)


#----------building and writing the star_mult_arr----------------------------#  

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

# star-wise list of multiplets                                                     
star_mult_arr = create_synthetic_mults.get_star_mult_arr(GVARS)
dd.io.save(f'{metadata_path}/star_mult_arr.h5', star_mult_arr)
