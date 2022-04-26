from collections import namedtuple
import jax.tree_util as tu
import time
from tqdm import tqdm
from jax import jit
from jax.lax import cond as cond
import jax.numpy as jnp
import numpy as np
import pickle

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def create_namedtuple(tname, keys, values):
    NT = namedtuple(tname, keys)
    nt = NT(*values)
    return nt

def tree_map_CNM_AND_NBS(CNM_AND_NBS):
    # converting to tuples and nestes tuples for easy of passing
    nl_nbs = tuple(map(tuple, CNM_AND_NBS.nl_nbs))
    nl_nbs_idx = tuple(CNM_AND_NBS.nl_nbs_idx)
    omega_nbs = tuple(CNM_AND_NBS.omega_nbs)

    CENMULT_AND_NBS = create_namedtuple('CENMULT_AND_NBS',
                                        ['nl_nbs',
                                         'nl_nbs_idx',
                                         'omega_nbs'],
                                        (nl_nbs,
                                         nl_nbs_idx,
                                         omega_nbs))

    return CENMULT_AND_NBS

def tree_map_SUBMAT_DICT(SUBMAT_DICT):
    # converting to nested tuples first
    SUBMAT_DICT = tu.tree_map(lambda x: tuple(map(tuple, x)), SUBMAT_DICT)

    return SUBMAT_DICT

def jax_Omega(ell, N):
    """Computes Omega_N^\ell"""
    return cond(abs(N) > ell,
                lambda __: 0.0,
                lambda __: jnp.sqrt(0.5 * (ell+N) * (ell-N+1)),
                operand=None)
    
def jax_minus1pow_vec(num):
    """Computes (-1)^n"""
    modval = num % 2
    return (-1)**modval

def jax_gamma(ell):
    """Computes gamma_ell"""
    return jnp.sqrt((2*ell + 1)/4/jnp.pi)

def mults2modes(mult_arr):
    """Computes the modes from the multiplet array.
    """
    Nmodes = np.sum(2 * mult_arr[:,1] + 1)
    
    mode_arr = np.zeros((3, Nmodes), dtype='int')
    
    # to keep track of filling index
    index_counter = 0
    
    for mult_ind in range(len(mult_arr)):
        n = mult_arr[mult_ind, 0]
        ell = mult_arr[mult_ind, 1]
        m_arr = np.arange(-ell, ell+1, dtype='int')
        
        mode_arr[0, index_counter: index_counter + 2*ell+1] = n
        mode_arr[1, index_counter: index_counter + 2*ell+1] = ell
        mode_arr[2, index_counter: index_counter + 2*ell+1] = m_arr
        
        index_counter += 2*ell + 1
        
    return mode_arr

def get_mult_freqs(GVARS, star_mult_arr):
    """This functions returns the unperturbed frequencies of 
    modes in the Nmodes x 1 form which can be directly added the freq splittings.
    """
    
    # will omit the first index at the end
    mode_unpert_freqs = np.array([0])
    
    for star_key in star_mult_arr.keys():
        mult_this_Stype = star_mult_arr[f'{star_key}'][:, 1:]
        
        for mult_ind in range(len(mult_this_Stype)):
            n0, ell0 = mult_this_Stype[mult_ind]
            nl_idx = GVARS.nl_all_list.index([n0, ell0])

            # the array of (2*ell0 + 1) unperturbed omegas for a multiplet with ell0
            omega4mult = np.ones(2 * ell0 + 1) * GVARS.omega_list[nl_idx]
            
            mode_unpert_freqs = np.append(mode_unpert_freqs, omega4mult)
        
        
    # omitting the first invalid entry
    mode_unpert_freqs = mode_unpert_freqs[1:]
    
    # converting to muHz 
    mode_unpert_freqs = mode_unpert_freqs * GVARS.OM * 1e6
    
    return mode_unpert_freqs
        
