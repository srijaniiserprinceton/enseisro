# import jax.numpy as np
import numpy as np
import enseisro.loading_functions as loadfunc
import enseisro.misc_functions as FN
from scipy.integrate import simps
import sys
NAX = np.newaxis

# {{{ def compute_kernel():                                                                                                                                                           
def compute_kernel(GVAR, mult, rcz, s_arr, Nparams = 2):
        """Function to compute the frequency splittings                                                                                                                                  
        under isolated multiplet approximation.                                                                                                                                          
        - GVAR: Dictionary of global parameters                                                                                                                                          
        - mult: Isolated multiplet whose splitting is to be calculated                                                                                                                   
        """
        n, ell = mult[0][0], mult[0][1]
        m = np.arange(-ell, ell+1)

        # locating index in radius corresponding to rcz
        rcz_ind = np.argmin(np.abs(GVAR.r - rcz))

        # mult_idx = FN.nl_idx(GVAR, n, ell)
        # omeganl = GVAR.omega_list[mult_idx]
        
        # shape (m x s)
        wigvals = np.zeros((2*ell+1, len(s_arr)))
        for i in range(len(s_arr)):
            wigvals[:, i] = FN.w3j_vecm(ell, s_arr[i], ell, -m, 0*m, m)

        prod_gammas = FN.gamma(ell) * FN.gamma(ell)  # * FN.gamma(s_arr), not using for Omega

        Tsr = compute_Tsr(GVAR, mult, mult, s_arr)   # shape (s x r)

        # summing over the radial grid to do K_{ij} F_{jk}. See notes in Overleaf
        # the idea is that since F_{jk} are step functions going from 0 to 1,
        # it implies a summation of the kernels before and after the rcz

        K = np.zeros((len(s_arr), Nparams))    # shape (s x Nparams)

        # writing this for just two params for the step function case
        K[:,0] = simps(Tsr[:,:rcz_ind], axis=1, x=GVAR.r[:rcz_ind])
        K[:,1] = simps(Tsr[:,rcz_ind:], axis=1, x=GVAR.r[rcz_ind:])

        # constructing the other prefactors. Shape (m x s)
        prefactors = wigvals * (FN.minus1pow_vec(m) * 4*np.pi * prod_gammas)[:,NAX]

        # multiplying the various other factors. K shape is now (m x s x Nparams)
        K = K[NAX,:,:] *  prefactors[:,:,NAX]

        # shape (m x s x Nparams)
        return K
# }}} def compute_kernel()



# {{{ def compute_kernel_for_Delta_Omega():                                                                                                                                                           
def compute_kernel_for_Delta_Omega(GVAR, mult, rcz, s_arr, Nparams = 2):
        """Function to compute the frequency splittings                                                                                                                                  
        under isolated multiplet approximation.                                                                                                                                          
        - GVAR: Dictionary of global parameters                                                                                                                                          
        - mult: Isolated multiplet whose splitting is to be calculated                                                                                                                   
        """
        n, ell = mult[0][0], mult[0][1]
        m = np.arange(-ell, ell+1)

        # locating index in radius corresponding to rcz
        rcz_ind = np.argmin(np.abs(GVAR.r - rcz))

        # mult_idx = FN.nl_idx(GVAR, n, ell)
        # omeganl = GVAR.omega_list[mult_idx]
        
        # shape (m x s)
        wigvals = np.zeros((2*ell+1, len(s_arr)))
        for i in range(len(s_arr)):
            wigvals[:, i] = FN.w3j_vecm(ell, s_arr[i], ell, -m, 0*m, m)

        prod_gammas = FN.gamma(ell) * FN.gamma(ell)  # * FN.gamma(s_arr), not using for Omega

        Tsr = compute_Tsr(GVAR, mult, mult, s_arr)   # shape (s x r)

        # summing over the radial grid to do K_{ij} F_{jk}. See notes in Overleaf
        # the idea is that since F_{jk} are step functions going from 0 to 1,
        # it implies a summation of the kernels before and after the rcz

        K = np.zeros((len(s_arr), Nparams))    # shape (s x Nparams)

        # writing this for just two params for the step function case
        # since \Omega_j = \Omega_{out} (F_{j1} + F_{j2}) + \Delta \Omega F_{j1}
        K[:,0] = simps(Tsr[:,:], axis=1, x=GVAR.r[:])
        K[:,1] = simps(Tsr[:,:rcz_ind], axis=1, x=GVAR.r[:rcz_ind])

        # constructing the other prefactors. Shape (m x s)
        prefactors = wigvals * (FN.minus1pow_vec(m) * 4*np.pi * prod_gammas)[:,NAX]

        # multiplying the various other factors. K shape is now (m x s x Nparams)
        K = K[NAX,:,:] *  prefactors[:,:,NAX]

        # shape (m x s x Nparams)
        return K
# }}} def compute_kernel_for_Delta_Omega()



# {{{ compute_Tsr():                                                                                                                                                                     
def compute_Tsr(GVAR, mult1, mult2, s_arr):
        """Function to compute the T-kern as in LR92 for coupling                                                                                                                        
        of two multiplets.                                                                                                                                                               
        """
        Tsr = np.zeros((len(s_arr), len(GVAR.r)))

        # reading off the ells corresponding to the multiplets                                                                                                                           
        ell1, ell2 = mult1[0][1], mult2[0][1]

        # allowing for different multiplets for mode-coupling purposes                                                                                                                   
        m1idx = FN.nl_idx_vec(GVAR, mult1)[0]
        m2idx = FN.nl_idx_vec(GVAR, mult2)[0]

        # loading the eigenfunctions                                                                                                                                                     
        U1, V1 = loadfunc.get_eig(GVAR, m1idx)
        U2, V2 = loadfunc.get_eig(GVAR, m2idx)

        L1sq = ell1*(ell1+1)
        L2sq = ell2*(ell2+1)

        Om1 = FN.Omega(ell1, 0)
        Om2 = FN.Omega(ell2, 0)
        for i in range(len(s_arr)):
            s = s_arr[i]
            ls2fac = L1sq + L2sq - s*(s+1)
            eigfac = U2*V1 + V2*U1 - U1*U2 - 0.5*V1*V2*ls2fac
            wigval = FN.w3j(ell1, s, ell2, -1, 0, 1)
            Tsr[i, :] = -(1 - FN.minus1pow(ell1 + ell2 + s)) * \
                Om1 * Om2 * wigval * eigfac   # / GVAR.r, not using for Omega                                                                                                            

        return Tsr
# }}} compute_Tsr()   
