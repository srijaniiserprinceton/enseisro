# import jax.numpy as np
import numpy as np
import enseisro.forward_functions_Omega as forfunc_Om
from enseisro import get_kernels as get_kerns

NAX = np.newaxis

def make_A(GVAR, modes, sigma_arr, rcz=0.7, Nparams=2, smax=1):
    """This function is used to create the A matrix in
    A . a = d for our linear inverse problem
    """
    # B = K_{ij} F_{jk} as defined in the Overleaf notes
    Nmodes = modes.shape[1]
    B_arr = np.zeros((Nmodes, Nparams))

    # creating s_arr
    s_arr = np.arange(1, smax+1, 2)
    
    # indices of where to start and end in mode-labels when filling B_{ik}
    modefill_start_ind, modefill_end_ind = 0, 0

    i = 0
    while(i < Nmodes):
        # extracting the m's from the instantaneous multiplet                                                                                                                           
        n_inst, ell_inst = modes[0,i], modes[1,i]
        
        # array that contains all the m's for this instantaneous multiplet                                                                                                              
        inst_mult_marr = get_inst_mult_marr(n_inst, ell_inst, modes)
        
        # creating the mode end index                                                                                                                                                   
        modefill_end_ind = modefill_start_ind + len(inst_mult_marr)
        
        # getting the K_{ij}F_{jk}
        KF = get_kerns.compute_kernel(GVAR, np.array([[n_inst, ell_inst]]), rcz, s_arr)    # shape (m x s x Nparams)

        # for now only one s. So, I am doing this explicitly
        KF = KF[:,0,:]      # shape (m x Nparams)

        B_arr[modefill_start_ind:modefill_end_ind,:] = KF[inst_mult_marr + ell_inst,:]

        # moving onto the next multiplet
        i += len(inst_mult_marr)

        # moving the mode start index to the next multiplet
        modefill_start_ind = modefill_end_ind

    # finally we divide by the respective freq splitting uncertainties
    A = B_arr / sigma_arr[:,NAX]

    return A
        

def make_d_synth(GVAR, modes, sigma_arr, Omega_synth=np.array([])):
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
        
        # array that contains all the m's for this instantaneous multiplet
        inst_mult_marr = get_inst_mult_marr(n_inst, ell_inst, modes)
        
        # computing the forward problem to get the mx1 array of frequency splittings
        domega_m = forfunc_Om.compute_splitting(GVAR, Omega_synth[0], np.array([[n_inst,ell_inst]]))

        # appending the necessary entries of m in d
        m_indices = inst_mult_marr + ell_inst
        d = np.append(d, domega_m[m_indices])
        
        # moving onto the next multiplet
        i += len(inst_mult_marr)
    
    # scaling by the uncertainty in freq splitting
    d = d / sigma_arr
    
    return d


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
