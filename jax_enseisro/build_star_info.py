import numpy as np

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

num_startype_arr = np.array([4, 2, 8, 5], dtype='int')

nmin, nmax = 16, 24
lmin, lmax = 1, 3

rand_mults = 1

# Max angular degree for Omega_s
smax = 3

# whether to use the noise model to add synthetic noise                                    
# to the frequency splitting data
add_Noise = 0

# whether to use (Omega_in, Omega_out) or (Omega_out, Delta_Omega)
use_Delta = 1

#------writing out the metadata to be used in other files-------------------#

with open("inversion_metadata/.star_metadata.dat", "w") as f:
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
np.save('inversion_metadata/num_startype_arr.npy', num_startype_arr)
