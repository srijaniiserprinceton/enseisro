import numpy as np

from jax_enseisro import globalvars as gvars_jax
from data_scripts import create_synthetic_mults

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
star_mult_arr = create_synthetic_mults.get_star_mult_arr(GVARS)



