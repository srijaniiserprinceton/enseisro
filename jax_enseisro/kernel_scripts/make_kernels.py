import numpy as np

from jax_enseisro.setup_scripts import misc_functions as misc_fn
from jax_enseisro.setup_scripts import build_cenmults as build_cnm
from jax_enseisro.setup_scripts import sparse_precompute as precompute
from jax_enseisro.setup_scripts import prune_multiplets
from jax_enseisro.setup_scripts import load_multiplets

def make_kernels(star_mult_arr, GVARS):
    # dictionary of kernels to be filled by Star Type
    kernels = {}

    # the rcz of stars in each Star Type                                                   
    rcz_startype_arr = np.load(f'{GVARS.metadata}/rcz_startype_arr.npy')
    
    # looping over the different Star Types
    for startype_label in range(GVARS.nStype):
        # the location of stardata for this type
        star_data = f'{GVARS.star_data_paths[startype_label]}/data_files'
        star_eigdir = f'{GVARS.star_data_paths[startype_label]}/eig_files'
        
        # the model specific information
        r = np.loadtxt(f"{star_data}/r.dat")
        rcz = rcz_startype_arr[startype_label]

        #-------------slicing radius appropriately-------------#
        rmin_idx = GVARS.get_idx(r, GVARS.rmin)
        # removing the grid point corresponding to r=0                                        
        # because Tsr has 1/r factor                                                          
        if GVARS.rmin == 0:
            rmin_idx += 1
            
        rmax_idx = GVARS.get_idx(r, GVARS.rmax)
        
        # slicing r appropriately
        r = r[rmin_idx:rmax_idx]

        # getting rthe rcz index in the sliced radius grid
        rcz_idx = GVARS.get_idx(r, rcz)
        #-----------------------------------------------------#

        nl_all = np.loadtxt(f"{star_data}/nl.dat").astype('int')
        omega_list = np.loadtxt(f"{star_data}/muhz.dat") * 1e-6 / GVARS.OM
        
        # extracting the multiplets for hte star type
        startype_mults = star_mult_arr[f'{startype_label}'][:, 1:]
        n0_arr, ell0_arr = startype_mults[:, 0], startype_mults[:, 1]

        # making a namedtuple GVARS_startype for the specific background model
        GVARS_startype = misc_fn.create_namedtuple('GVARS_STARTYPE',
                                                   ['Stype',
                                                    'n0_arr',
                                                    'ell0_arr',
                                                    'r',
                                                    'nl_all',
                                                    'omega_list',
                                                    'eigdir',
                                                    'rmin_idx',
                                                    'rmax_idx',
                                                    's_arr'],
                                                   (startype_label,
                                                    n0_arr,
                                                    ell0_arr,
                                                    r,
                                                    nl_all,
                                                    omega_list,
                                                    star_eigdir,
                                                    rmin_idx,
                                                    rmax_idx,
                                                    GVARS.s_arr))

        # getting the cenmult namedtuple for the mults in this startype
        cenmult_this_startype = build_cnm.getnt4cenmult(GVARS_startype)
        
        # extracting attributes from CNM_AND_NBS                                              
        num_cnm = len(cenmult_this_startype.omega_cnm)
        ellmax = np.max(cenmult_this_startype.nl_cnm[:,1])

        # getting pruned attributes
        nl_pruned, nl_idx_pruned, omega_pruned, wig_list, wig_idx =\
            prune_multiplets.get_pruned_attributes(GVARS_startype,
                                                   cenmult_this_startype)

        lm = load_multiplets.load_multiplets(GVARS_startype, nl_pruned,
                                             nl_idx_pruned,
                                             omega_pruned)
        
        
        # the dictionary of elements that are needed in precomputation
        precomp_dict = misc_fn.create_namedtuple('PRECOMP_DICT',
                                                 ['Stype',
                                                  'num_cnm',
                                                  'ellmax',
                                                  'wig_list',
                                                  'wig_idx',
                                                  'nl_idx_pruned',
                                                  'lm',
                                                  'r',
                                                  's_arr',
                                                  'rcz_idx'],
                                                 (startype_label,
                                                  num_cnm,
                                                  ellmax,
                                                  wig_list,
                                                  wig_idx,
                                                  nl_idx_pruned,
                                                  lm,
                                                  r,
                                                  GVARS.s_arr,
                                                  rcz_idx))
    
        kernels_this_startype =\
            precompute.build_kernels_all_cenmults(cenmult_this_startype,
                                                  precomp_dict)
                                                 
        kernels[f'{startype_label}'] = kernels_this_startype

    return kernels
