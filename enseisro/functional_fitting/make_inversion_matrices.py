# import jax.numpy as np
import numpy as np
import enseisro.forward_functions_Omega as forfunc_Om
from enseisro import get_kernels as get_kerns

def make_A(GVAR, modes, Nparams=2, smax=1):
    """This function is used to create the A matrix in
    A . a = d for our linear inverse problem
    """
    # B = K_{ij} F_{jk} as defined in the Overleaf notes
    B_arr = np.zeros((len(modes), Nparams))

    for i in modes

def make_d(GVAR, modes, use_synth=True, Omega_synth=np.array([])):
    """This function is used to build the data vector d  of frequency splittings
    in the forward problem A . a = d
    """
    d = np.array([])   # creating the empty data (frequency splitting) vector
    
    # extracting the number of modes present
    Nmodes = modes.shape[1]
    
    i = 0
    while(i < Nmodes):
        # extracting the m's from the instantaneous multiplet
        n_inst, ell_inst = modes[0,i], modes[1,i]
        
        # extracting all the available m's for this multiplet
        mask_inst_mult = (modes[0,:]==n_inst)*(modes[1,:]==ell_inst)
        # array that contains all the m's for this instantaneous multiplet
        inst_mult_marr = modes[:,mask_inst_mult][2,:]
        
        # computing the forward problem to get the mx1 array of frequency splittings
        domega_m = forfunc_Om.compute_splitting(GVAR, Omega_synth,np.array([[n_inst,ell_inst]]))

        # appending the necessary entries of m in d
        m_indices = inst_mult_marr + ell_inst
        d = np.append(d, domega_m[m_indices])
        
        # moving onto the next multiplet
        i += len(inst_mult_marr)
    
    
    return d
