"""Misc functions needed for the module"""
import argparse

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
