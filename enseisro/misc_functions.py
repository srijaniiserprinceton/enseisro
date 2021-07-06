"""Misc functions needed for the module"""
import argparse
# import jax.numpy as np
import numpy as np
import py3nj

# {{{ def create_argparser():                                                                                                                                                            
def create_argparser():
    """Creates argument parser for arguments passed during                                                                                                                               
    execution of script.                                                                                                                                                                 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--n0", help="radial order", type=int)
    parser.add_argument("--l0", help="angular degree", type=int)
    parser.add_argument("--lmin", help="min angular degree", type=int)
    parser.add_argument("--lmax", help="max angular degree", type=int)
    parser.add_argument("--maxiter", help="max MCMC iterations",
                        type=int)
    parser.add_argument("--precompute",
                        help="precompute the integrals upto r=0.9",
                        action="store_true")
    parser.add_argument("--use_precomputed",
                        help="use precomputed integrals",
                        action="store_true")
    parser.add_argument("--usempi",
                        help='use MPI for Bayesian estimation',
                        action="store_true")
    parser.add_argument("--parallel",
                        help='parallel processing',
                        action="store_true")
    args = parser.parse_args()
    return args
# }}} create_argparser() 

# {{{ def nl_idx():
def nl_idx(GVAR, n0, l0):
    try:
        idx = GVAR.nl_all_list.index([n0, l0])
    except ValueError:
        idx = None
        logger.error('Mode not found')
    return idx
# }}} nl_idx()

# {{{ def nl_idx_vec():
def nl_idx_vec(GVAR, nl_list):
    nlnum = nl_list.shape[0]
    nlidx = np.zeros(nlnum, dtype=np.int32)
    for i in range(nlnum):
        nlidx[i] = nl_idx(GVAR, nl_list[i][0],
                               nl_list[i][1])
    return nlidx
# }}} nl_idx_vec()

# {{{ def w3j():
def w3j(l1, l2, l3, m1, m2, m3):
    l1 = int(2*l1)
    l2 = int(2*l2)
    l3 = int(2*l3)
    m1 = int(2*m1)
    m2 = int(2*m2)
    m3 = int(2*m3)
    try:
        wigval = py3nj.wigner3j(l1, l2, l3, m1, m2, m3)
    except ValueError:
        return 0.0
    return wigval
# }}} w3j()

# {{{ def w3j_vecm():
def w3j_vecm(l1, l2, l3, m1, m2, m3):
    l1 = int(2*l1)
    l2 = int(2*l2)
    l3 = int(2*l3)
    m1 = 2*m1
    m2 = 2*m2
    m3 = 2*m3
    wigvals = py3nj.wigner3j(l1, l2, l3, m1, m2, m3)
    return wigvals
# }}} def w3j_vecm()

# {{{ def Omega():
def Omega(ell, N):
    if abs(N) > ell:
        return 0
    else:
        return np.sqrt(0.5 * (ell+N) * (ell-N+1))
# }}} Omega()

# {{{ def minus1pow():
def minus1pow(num):
    if num%2 == 1:
        return -1.0
    else:
        return 1.0
# }}} minus1pow()

# {{{ minus1pow_vec():
def minus1pow_vec(num):
    modval = num % 2
    retval = np.zeros_like(modval)
    retval[modval == 1] = -1.0
    retval[modval == 0] = 1.0
    return retval
# }}} minus1pow_vec()

# {{{ def gamma():
def gamma(ell):
    return np.sqrt((2*ell + 1)/4/np.pi)
# }}} gamma()


# {{{ def reshape_KF():
def reshape_KF(KF):
    """This function reshapes KF adequately to tile the 
    s dimensions as extra set of columns. So, KF is 
    converted from shape (Nmodes x Nparams x s) tp the shape
    (Nmodes x (Nparams*s))
    """
    return np.reshape(KF,(KF.shape[0],-1),'F')
# }}} def reshape_KF()

# {{{ def build_mults():
def build_mults(nmin, nmax, lmin, lmax):
    """This function creates the list of multiplets
    given the nmin, nmax, lmin and lmax
    """
    Nmults = (nmax - nmin + 1) * (lmax - lmin + 1)
    mults = np.zeros((Nmults,2),dtype='int')
    
    mult_count = int(0)
    for n_ind, n in enumerate(range(nmin, nmax+1)):
        for ell_ind, ell in enumerate(range(lmin, lmax+1)):
            mults[mult_count,:] = np.array([n,ell],dtype='int')
            mult_count += 1

    return mults

# }}} def build_mults()

# def format_terminal_output():
def format_terminal_output(synthetic, inverted, error):
    """This function formats the output to be printed in the terminal
    """
    dash = '-' * 95
    col_headers = np.array(['in nHz', 'Om_in_1', 'Om_out_1', 'Om_in_3', 'Om_out_3'])
    row_headers = np.array(['Synthetic', 'Inverted', 'Error(+/-)'])
    
    for row_ind in range(4):
        if(row_ind == 0):
            print(dash)
            print('{:<20s}{:^20s}{:^20s}{:^20s}{:^20s}'.format(col_headers[0],\
                col_headers[1],col_headers[2],col_headers[3],col_headers[4]))
            print(dash)
        elif(row_ind == 1):
            print('{:<20s}{:^20f}{:^20f}{:^20f}{:^20f}'.format(row_headers[row_ind-1],\
                        synthetic[0],synthetic[1],synthetic[2],synthetic[3]))
        elif(row_ind == 2):
            print('{:<20s}{:^20f}{:^20f}{:^20f}{:^20f}'.format(row_headers[row_ind-1],\
                        inverted[0],inverted[1],inverted[2],inverted[3]))
        else:
            print('{:<20s}{:^20f}{:^20f}{:^20f}{:^20f}'.format(row_headers[row_ind-1],\
                        error[0],error[1],error[2],error[3]))

# }}} def format_terminal_output()

# {{{ def build_star_labels_and_all_modes():
def build_star_labels_and_all_modes(Nstars, modes_single_star):
    Nmodes_single_star = modes_single_star.shape[1]
    star_label_arr = np.zeros(Nstars * Nmodes_single_star, dtype='int')
    modes = np.zeros((3, len(star_label_arr)), dtype='int')
                     
    # filling in labels and repeating the modes
    all_mode_fill_start_ind, all_mode_fill_end_ind = 0, Nmodes_single_star
    for i in range(Nstars):
        star_label_arr[all_mode_fill_start_ind:all_mode_fill_end_ind] = i 
        # filling in the same set of modes for all the stars
        modes[:,all_mode_fill_start_ind:all_mode_fill_end_ind] = modes_single_star
    
        # updating the start and end indices
        all_mode_fill_start_ind += Nmodes_single_star
        all_mode_fill_end_ind += Nmodes_single_star

    return star_label_arr, modes
