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
