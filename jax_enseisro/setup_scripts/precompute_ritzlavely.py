from ritzLavelyPy import rlclass as RLC
from jax_enseisro import globalvars as gvar_jax
import numpy as np
import argparse
import deepdish as dd
import os
import sys

current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)

metadata_path = f'{package_dir}/inversion_metadata'

#-----------------------------------------------------------------------
def gen_RL_poly():
    # the array of ells
    ell0_arr = np.arange(1, max_ell + 1)
    RL_poly = np.zeros((len(ell0_arr), jmax+1, 2*max_ell+1), dtype=np.float64)

    for ell_i, ell in enumerate(ell0_arr):
        # checking if jmax+1 <= 2*ell (since jmax=odd)
        if(jmax+1 <= 2*ell):
            jmax_new = jmax
            RLP = RLC.ritzLavelyPoly(ell, jmax_new)
        else:
            jmax_new = 2*ell - 1
            RLP = RLC.ritzLavelyPoly(ell, jmax_new)
        RL_poly[ell_i, :jmax_new+1, :2*ell+1] = RLP.Pjl
        
    # storing it as a dictionary with 's' keys
    RL_poly_dict = {}
    
    for ell_i, ell in enumerate(ell0_arr):
        RL_poly_dict[f'{ell}'] = {}
        for s in range(jmax+1):
            # checking if all the entries are zero, then its an
            # invalid jmax, ell combination. Settting all elements to 1
            # This is fine since the Wigners for these are 0's.
            # So, the wigval_acoeffs is zero and not nan.
            RL_Poly_ell_s = RL_poly[ell_i, s, :2*ell+1]
            if(np.sum(RL_Poly_ell_s**2)==0):
                RL_Poly_ell_s += 1.0
                
            RL_poly_dict[f'{ell}'][f'{s}'] = RL_Poly_ell_s
        
    return RL_poly_dict
#-----------------------------------------------------------------------

def find_max_ell_from_star_mult_arr():
    # star-wise list of multiplets                                                            
    star_mult_arr = dd.io.load(f'{metadata_path}/star_mult_arr.h5')
    
    star_keys = star_mult_arr.keys()

    # setting an unrealistic value to start with
    max_ell = -1
    for star_key in star_keys:
        this_star_mult_arr = star_mult_arr[star_key]
        
        this_star_max_ell = np.max(this_star_mult_arr[:,2])
        if(this_star_max_ell > max_ell):
            max_ell = this_star_max_ell
            
    return max_ell
        

if __name__ == '__main__':
    ARGS = np.loadtxt(f"{metadata_path}/.star_metadata.dat")
    GVARS = gvar_jax.GlobalVars(nStype=int(ARGS[0]),
                                nmin=int(ARGS[1]),
                                nmax=int(ARGS[2]),
                                lmin=int(ARGS[3]),
                                lmax=int(ARGS[4]),
                                smax=int(ARGS[5]),
                                rand_mults=int(ARGS[6]),
                                add_Noise=int(ARGS[7]),
                                use_Delta=int(ARGS[8]),
                                metadata_path = f'{metadata_path}',
                                is_acoeffs_kernel=(ARGS[9]))

    # to find the max ell for which to compute the RLPoly
    max_ell = find_max_ell_from_star_mult_arr()
    jmax = GVARS.smax
    
    RL_poly = gen_RL_poly()
    dd.io.save(f'{metadata_path}/RL_poly.h5', RL_poly)
