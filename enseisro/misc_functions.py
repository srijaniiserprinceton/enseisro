"""Misc functions needed for the module"""
import argparse
# import jax.numpy as np
import numpy as np
import py3nj
import matplotlib.pyplot as plt

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

# {{{ def build_mults_single_star():
def build_mults_single_star(nmin, nmax, lmin, lmax):
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

# }}} def build_mults_single_star()

# {{{ def build_star_labels_and_mults_modes():
def build_star_labels_and_mults_modes(Nstars, mults_single_star, modes_single_star):
    Nmodes_single_star = modes_single_star.shape[1]
    star_modelabel_arr = np.zeros(Nstars * Nmodes_single_star, dtype='int')
    
    # choosing some dummy multiplet since this is for synthetics only
    Nmults_single_star = mults_single_star.shape[0]
    star_multlabel_arr = np.zeros(Nstars * Nmults_single_star, dtype='int')

    modes = np.zeros((3, len(star_modelabel_arr)), dtype='int')
    mults = np.zeros((len(star_multlabel_arr), 2), dtype='int')
                     
    # filling in labels and repeating the modes
    all_mode_fill_start_ind, all_mode_fill_end_ind = 0, Nmodes_single_star
    all_mult_fill_start_ind, all_mult_fill_end_ind = 0, Nmults_single_star

    for i in range(Nstars):
        star_modelabel_arr[all_mode_fill_start_ind:all_mode_fill_end_ind] = i 
        star_multlabel_arr[all_mult_fill_start_ind:all_mult_fill_end_ind] = i

        # filling in the same set of modes for all the stars
        modes[:,all_mode_fill_start_ind:all_mode_fill_end_ind] = modes_single_star
        mults[all_mult_fill_start_ind:all_mult_fill_end_ind,:] = mults_single_star

        # updating the start and end indices
        all_mode_fill_start_ind += Nmodes_single_star
        all_mode_fill_end_ind += Nmodes_single_star

        # updating the start and end indices                                                                                                                                                     
        all_mult_fill_start_ind += Nmults_single_star
        all_mult_fill_end_ind += Nmults_single_star

    return star_multlabel_arr, mults, star_modelabel_arr, modes

# }}} def build_star_labels_and_mults_modes()

def get_inst_mult_marr(n_inst, ell_inst, modes):
    """This function takes the instantaneous n and ell for a mode in the                                                                                                                        
    list of all observed modes and returns the m's corresponding to that                                                                                                                         
    (n_inst, ell_inst) multiplet. This is to avoid repeated kernel computation for                                                                                                               
    each m which is clearly unnecessary."""

    # extracting all the available m's for this multiplet                                                                                                                                      
    mask_inst_mult = (modes[0,:]==n_inst)*(modes[1,:]==ell_inst)
    # array that contains all the m's for this instantaneous multiplet                                                                                                                         
    inst_mult_marr = modes[:,mask_inst_mult][2,:]

    return inst_mult_marr



# {{{ def get_omega_nlm_from_delta_omega_nlm():
def get_omega_nlm_from_delta_omega_nlm(GVAR, delta_omega_nlm_arr, Nstars, mults, modes, 
                                       star_multlabel_arr, star_modelabel_arr):
    """Converts the frequency splittings detla omega_nlm of respective modes to the 
    total frequency omega_nlm.
    
    Parameters
    ----------
    delta_omega_nlm_arr : array_like, float
        Frequency splittings arranged according to the global modes array in non-dimensional units.
    Nstars : int
        Total number of stars in the ensemble.
    mults : numpy.ndarray, int
        Array containing the multiplets used  across all stars.
    modes : numpy.ndarray, int
        Array containing the modes used across all the stars.
    star_multlabel_arr : array_like, int
        Labelling the multiplets according to which star it belongs to.
    star_modelabel_arr : array_like, int
        Labelling the modes according to which star it belongs to.
    """

    mult_idx = nl_idx_vec(GVAR, mults)
    omeganl = GVAR.omega_list[mult_idx]
    
    
    # to store the omega_nl for all modes                                                                                                                         
    omeganl_arr = np.zeros(modes.shape[1])
    
    # looping over the multiplets and storing                                                                                            
    current_index  = 0  # keeping a counter on index position                                                                                                                    
    
    for i in range(Nstars):
        mult_star_ind = (star_multlabel_arr == i)
        mult_star = mults[mult_star_ind]
        for mult_ind, mult in enumerate(mult_star):
            n_inst, ell_inst = mult[0], mult[1]
            modes_current_star = modes[:,star_modelabel_arr==i]
            inst_mult_marr = get_inst_mult_marr(n_inst, ell_inst, modes_current_star)
            Nmodes_in_mult = len(inst_mult_marr)
            omeganl_arr[current_index:current_index + Nmodes_in_mult] = omeganl[mult_ind]
            current_index += Nmodes_in_mult
            
            
    # total freq = omeganl + delta omega_nlm                                                                                                                      
    omega_nlm_arr = omeganl_arr + delta_omega_nlm_arr

    # we need to pass the mode_freq_arr in muHz                                                                                                                                 
    omega_nlm_arr *= (GVAR.OM * 1e6)
    
    return omega_nlm_arr

# }}} def get_omega_nlm_from_delta_omega_nlm()


# {{{ def gen_freq_splitting_noise():
def gen_freq_splitting_noise(sigma_arr):
    """Generates gaussian noise centered at zero
    with standard deviation given by sigma_arr.

    Parameters
    ----------
    sigma_arr : array_like, float
        Array of standard deviations.
    """
    mean = np.zeros_like(sigma_arr)
    delta_omega_nlm_noise = np.random.normal(loc=mean, scale=sigma_arr)
    
    return delta_omega_nlm_noise

# }}} def gen_freq_splitting_noise()


# {{{ def propagate_errors():
def propagate_errors(avg_DOmega_by_Omega, Omega_avg, DOmega, err_Omega_avg, err_DOmega):
    """Carries out error propagation from the errors of 
    Omega and Delta Omega to the error for Delta Omega / Omega.

    Parameters
    ----------
    Omega : float 
        The inverted Omega_out for averaged across stars in the ensemble.
    DOmega : float
        The inverted Delta Omega across all stars in the ensemble.
    err_Omega_avg : float
        The error in Omega_out averaged across stars in the ensemble.
    err_DOmega : float
        The error in Delta Omega inferred collectively across all stars.
    """
    
    # the average DOmega / Omega
    # DOmega_by_Omega_avg = DOmega / Omega_avg
    DOmega_by_Omega_avg = avg_DOmega_by_Omega

    # computing the error for DOmega / Omega using error propagation
    err_DOmega_by_Omega_avg = DOmega_by_Omega_avg * np.sqrt((err_DOmega/DOmega)**2 +\
                                            (err_Omega_avg/Omega_avg)**2)

    return DOmega_by_Omega_avg, err_DOmega_by_Omega_avg

# }}} def propagate_errors()

# {{{ def plot sigma_vs_freq():                                                                                                                                                          
def plot_sigma_vs_freq(sigma_arr, omega_nlm_arr):
    """Plots the uncertainty obtained from the                                                                                                                                           
    the noise model vs the mode frequencies.                                                                                                                                             
                                                                                                                                                                                         
    Parameters                                                                                                                                                                           
    ----------                                                                                                                                                                           
    sigma_arr : array_like, float                                                                                                                                                        
        Array containing the noise in nHz for all modes.                                                                                                                                 
    omega_nlm_arr : array_like, float                                                                                                                                                    
        Array containing omega_nlm in muHz for all modes.                                                                                                                                
    """
    sort_args_mode_freq = np.argsort(omega_nlm_arr)
    sigma_arr = sigma_arr[sort_args_mode_freq]
    omega_nlm_arr = omega_nlm_arr[sort_args_mode_freq]

    plt.plot(omega_nlm_arr, sigma_arr, 'k+', markersize=10)
    plt.plot(omega_nlm_arr, sigma_arr, 'k')

    plt.xlabel('$\omega_{nlm}$ in $\mu$Hz')
    plt.ylabel('$\sigma(\delta \omega_{nlm}$) in nHz')

    plt.savefig('./plotter/sigma_vs_freq.pdf')

# }}} def plot_sigma_vs_freq()   
