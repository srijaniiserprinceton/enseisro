import numpy as np
import os
import sys

# parent directory path of "tests/"                                                      
current_dir = os.path.dirname(os.path.realpath(__file__))
tests_dir = os.path.dirname(current_dir)

# bulding star info
os.system(f'python {current_dir}/build_star_info.py')


from jax_enseisro import globalvars as gvars_jax
from jax_enseisro.data_scripts import create_synthetic_mults
from jax_enseisro.inversion_scripts import make_kernels

def test_splittings_forward_problem():    
    # just to get access to the location of metadata
    metadata_path = f'{current_dir}'
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
    star_mult_arr = create_synthetic_mults.get_star_mult_arr(GVARS)
    
    # get data vector
    # data_vector = make_data_vector.get_d(star_mult_arr, GVARS)
    kernel = make_kernels.make_kernels(star_mult_arr, GVARS)
    
    # Omega_step in the parent drectory tests/
    Omega_step = np.load(f'{tests_dir}/Omega_step.npy')
    
    rcz_ind = np.argmin(np.abs(GVARS.r - 0.7))
    
    Omega_1_in = Omega_step[0,rcz_ind-1]
    delta_Omega_1 = Omega_step[0, rcz_ind] - Omega_step[0, rcz_ind-1]
    delta_Omega_3 = Omega_step[1, rcz_ind]
    
    model_params = np.array([Omega_1_in, delta_Omega_1, delta_Omega_3])
    
    # frequency splitting
    freq_split_nondim = model_params @ kernel['0']
    
    # frequency splitting in nHz
    freq_split = freq_split_nondim * GVARS.OM * 1e9
    
    # reference frequency that I had checked against qdPy
    # this was not an exact match but was pretty close. 
    # I reduced rcz and the error with qdPy reduced. So it might be because 
    # of how a discrete step function behaves as compared to a continuous function
    # in an integral sense
    freq_split_ref = np.array([-868.58979428,
                               -414.42921892,
                               0.,
                               414.42921892,
                               868.58979428])
    
    np.testing.assert_array_almost_equal(freq_split, freq_split_ref)
    
    # deleting the metadata files so that it doesn't interfere with next runs
    os.system(f'rm {current_dir}/*.npy {current_dir}/*.h5' +
              f' {current_dir}/.star_metadata.dat')

# for a standalone call to this testing
if __name__ == "__main__":
    test_splittings_forward_problem()
