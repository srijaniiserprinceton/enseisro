# import jax.numpy as np
import numpy as np
import enseisro.forward_functions_Omega as forfunc_Om
from enseisro import get_kernels as get_kerns
from enseisro import misc_functions as FN

NAX = np.newaxis

def make_A(GVAR, modes, sigma_arr, smax, rcz=0.7, Nparams=2):
    """This function is used to create the A matrix in
    A . a = d for our linear inverse problem
    """

    # creating s_arr
    s_arr = np.arange(1, smax+1, 2)
    lens = len(s_arr)

    # B = K_{ij} F_{jk} as defined in the Overleaf notes
    Nmodes = modes.shape[1]
    B_arr = np.zeros((Nmodes, Nparams*lens))    # shape (Nmodes x (Nparams*s))

    # indices of where to start and end in mode-labels when filling B_{ik}
    modefill_start_ind, modefill_end_ind = 0, 0

    i = 0
    while(i < Nmodes):
        # extracting the m's from the instantaneous multiplet                                                                                                                           
        n_inst, ell_inst = modes[0,i], modes[1,i]
        
        # array that contains all the m's for this instantaneous multiplet                                                                                                              
        inst_mult_marr = FN.get_inst_mult_marr(n_inst, ell_inst, modes)
        
        # creating the mode end index                                                                                                                                                   
        modefill_end_ind = modefill_start_ind + len(inst_mult_marr)
        
        # getting the K_{ij}F_{jk}
        KF = get_kerns.compute_kernel(GVAR, np.array([[n_inst, ell_inst]]), rcz, s_arr)    # shape (m x s x Nparams)

        # changing the shape to (m x Nparams x s)
        KF = np.swapaxes(KF,1,2)

        # for now only one s. So, I am doing this explicitly
        # KF = KF[:,:,0]      # shape (m x Nparams)

        # tiling appropriately to allow inversion for multiple s. This just means we 
        # append the s > 1 dimensions to Nparams col. So new shape is (m x (Nparams * s))
        # reshaping in fortran style so that s-dimensions are added to the cols
        # KF = np.reshape(KF,(KF.shape[0],-1),'F')
        KF = FN.reshape_KF(KF)

        B_arr[modefill_start_ind:modefill_end_ind,:] = KF[inst_mult_marr + ell_inst,:]

        # moving onto the next multiplet
        i += len(inst_mult_marr)

        # moving the mode start index to the next multiplet
        modefill_start_ind = modefill_end_ind

    # finally we divide by the respective freq splitting uncertainties
    A = B_arr / sigma_arr[:,NAX]

    return A   # shape (Nmodes x Nparams)
        


def make_A_for_Delta_Omega(GVAR, modes, sigma_arr, smax, rcz=0.7, Nparams=2):
    """This function is used to create the A matrix in
    A . a = d for our linear inverse problem
    """

    # creating s_arr
    s_arr = np.arange(1, smax+1, 2)
    lens = len(s_arr)

    # B = K_{ij} F_{jk} as defined in the Overleaf notes
    Nmodes = modes.shape[1]
    B_arr = np.zeros((Nmodes, Nparams*lens))    # shape (Nmodes x (Nparams*s))

    # indices of where to start and end in mode-labels when filling B_{ik}
    modefill_start_ind, modefill_end_ind = 0, 0

    i = 0
    while(i < Nmodes):
        # extracting the m's from the instantaneous multiplet                                                                                                                           
        n_inst, ell_inst = modes[0,i], modes[1,i]
        
        # array that contains all the m's for this instantaneous multiplet                                                                                                              
        inst_mult_marr = FN.get_inst_mult_marr(n_inst, ell_inst, modes)
        
        # creating the mode end index                                                                                                                                                   
        modefill_end_ind = modefill_start_ind + len(inst_mult_marr)
        
        # getting the K_{ij}F_{jk}
        KF = get_kerns.compute_kernel_for_Delta_Omega(GVAR, np.array([[n_inst, ell_inst]]), rcz, s_arr)    # shape (m x s x Nparams)

        # changing the shape to (m x Nparams x s)
        KF = np.swapaxes(KF,1,2)

        # for now only one s. So, I am doing this explicitly
        # KF = KF[:,:,0]      # shape (m x Nparams)

        # tiling appropriately to allow inversion for multiple s. This just means we 
        # append the s > 1 dimensions to Nparams col. So new shape is (m x (Nparams * s))
        # reshaping in fortran style so that s-dimensions are added to the cols
        # KF = np.reshape(KF,(KF.shape[0],-1),'F')
        KF = FN.reshape_KF(KF)

        B_arr[modefill_start_ind:modefill_end_ind,:] = KF[inst_mult_marr + ell_inst,:]

        # moving onto the next multiplet
        i += len(inst_mult_marr)

        # moving the mode start index to the next multiplet
        modefill_start_ind = modefill_end_ind

    # finally we divide by the respective freq splitting uncertainties
    A = B_arr / sigma_arr[:,NAX]

    return A   # shape (Nmodes x Nparams)
        



def make_d_synth_from_function(GVAR, modes, sigma_arr, Omega_synth):
    """This function is used to build the data vector d  of frequency splittings
    in the forward problem A . a = d
    """
    d = np.array([])   # creating the empty data (frequency splitting) vector
    
    # extracting the number of modes present. This should be all-stars, all modes
    Nmodes = modes.shape[1]
    
    i = 0
    while(i < Nmodes):
        # extracting the m's from the instantaneous multiplet
        n_inst, ell_inst = modes[0,i], modes[1,i]
        
        # array that contains all the m's for this instantaneous multiplet
        inst_mult_marr = FN.get_inst_mult_marr(n_inst, ell_inst, modes)
        
        # computing the forward problem to get the mx1 array of frequency splittings
        domega_m = forfunc_Om.compute_splitting_from_function(GVAR, Omega_synth, np.array([[n_inst,ell_inst]]))
        
        # appending the necessary entries of m in d
        m_indices = inst_mult_marr + ell_inst
        
        d = np.append(d, domega_m[m_indices])
        
        # moving onto the next multiplet
        i += len(inst_mult_marr)
        
    # scaling by the uncertainty in freq splitting
    d = d / sigma_arr
    
    return d



def make_d_synth_from_step_params(GVAR, modes_star, sigma_arr_star, Omega_step_params_star, rcz_ind_star, compute_freq_splitting_fn):
    """This function is used to build the data vector d  of frequency splittings
    in the forward problem A . a = d
    """
    d_star = np.array([])   # creating the empty data (frequency splitting) vector
    
    # extracting the number of modes present in current star
    Nmodes_star = modes_star.shape[1]


    i = 0
    while(i < Nmodes_star):
        # extracting the m's from the instantaneous multiplet
        n_inst, ell_inst = modes_star[0,i], modes_star[1,i]
        
        # array that contains all the m's for this instantaneous multiplet
        inst_mult_marr = FN.get_inst_mult_marr(n_inst, ell_inst, modes_star)

        # getting the star's label
        # star_label = star_label_arr[i]
        
        # getting the rcz_ind and the step_params from the particular star
        # rcz_ind = rcz_ind_arr[star_label]
        # Omega_step_params = Omega_step_params_arr[star_label]

        # computing the forward problem to get the mx1 array of frequency splittings
        domega_m = compute_freq_splitting_fn(GVAR, Omega_step_params_star,\
                                    np.array([[n_inst,ell_inst]]), rcz_ind_star)
        
        # appending the necessary entries of m in d
        m_indices = inst_mult_marr + ell_inst
        
        d_star = np.append(d_star, domega_m[m_indices])
        
        # moving onto the next multiplet
        i += len(inst_mult_marr)
        
    # scaling by the uncertainty in freq splitting
    d_star = d_star / sigma_arr_star
    
    return d_star
