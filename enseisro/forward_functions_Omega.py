# import jax.numpy as np
import numpy as np
import enseisro.loading_functions as loadfunc
import enseisro.misc_functions as FN
from scipy.integrate import simps

WFNAME = 'w_s/w.dat'

# {{{ def compute_splitting_from_function():
def compute_splitting_from_function(GVAR, Omegasr, mult):
        """Function to compute the frequency splittings
        under isolated multiplet approximation from the smooth profile.
        - GVAR: Dictionary of global parameters
        - mult: Isolated multiplet whose splitting is to be calculated
        """
        n, ell = mult[0][0], mult[0][1]
        m = np.arange(-ell, ell+1)

        # calculating the s_arr from w_s(r) shape
        lens = Omegasr.shape[0]
        s_arr = np.arange(1, (2*lens - 1) + 1, 2)

        # mult_idx = FN.nl_idx(GVAR, n, ell)
        # omeganl = GVAR.omega_list[mult_idx]

        wigvals = np.zeros((2*ell+1, len(s_arr)))
        for i in range(len(s_arr)):
            wigvals[:, i] = FN.w3j_vecm(ell, s_arr[i], ell, -m, 0*m, m)

        Tsr = compute_Tsr(GVAR, mult, mult, s_arr)
        
        # Loading the synthetic flow profile
        # -1 factor from definition of toroidal field                                                                                                                                   

        # wsr[0, :] *= 0.0 # setting w1 = 0                                                                                                                                            
 
        # wsr[1, :] *= 0.0 # setting w3 = 0                                                                                                                                              
        # wsr[2, :] *= 0.0 # setting w5 = 0                                                                                                                                              
        
        # integrating over radial grid
        # integrand = Tsr * wsr * (self.sup.gvar.rho * self.sup.gvar.r**2)[NAX, :]                                                                                                      
        integrand = Tsr * Omegasr   # since U and V are scaled by sqrt(rho) * r

        integral = simps(integrand, axis=1, x=GVAR.r)

        prod_gammas = FN.gamma(ell) * FN.gamma(ell)  # * FN.gamma(s_arr), not using for Omega

        # the factor of 2\omeganl has been taken off as compared to LR92 since we want \delta\omega
        Cvec = FN.minus1pow_vec(m) * 4*np.pi * (wigvals @ (prod_gammas * integral))

        return Cvec

# }}} compute_splitting_from_function()

# {{{ def compute_splitting_from_step_params():
def compute_splitting_from_step_params(GVAR, Omegas_step_params, mult, rcz_ind):
        """Function to compute the frequency splittings
        under isolated multiplet approximation from the step params.
        - GVAR: Dictionary of global parameters
        - mult: Isolated multiplet whose splitting is to be calculated
        """
        n, ell = mult[0][0], mult[0][1]
        m = np.arange(-ell, ell+1)

        # calculating the s_arr from w_s(r) shape
        lens = Omegas_step_params.shape[0]
        Nparams = Omegas_step_params.shape[1]

        s_arr = np.arange(1, (2*lens - 1) + 1, 2)

        # mult_idx = FN.nl_idx(GVAR, n, ell)
        # omeganl = GVAR.omega_list[mult_idx]

        wigvals = np.zeros((2*ell+1, len(s_arr)))
        for i in range(len(s_arr)):
            wigvals[:, i] = FN.w3j_vecm(ell, s_arr[i], ell, -m, 0*m, m)

        Tsr = compute_Tsr(GVAR, mult, mult, s_arr)
        
        # Loading the synthetic flow profile
        # -1 factor from definition of toroidal field                                                                                                                                   

        # wsr[0, :] *= 0.0 # setting w1 = 0                                                                                                                                            
 
        # wsr[1, :] *= 0.0 # setting w3 = 0                                                                                                                                              
        # wsr[2, :] *= 0.0 # setting w5 = 0                                                                                                                                              
        
        # integrating over radial grid
        # integrand = Tsr * wsr * (self.sup.gvar.rho * self.sup.gvar.r**2)[NAX, :]                                                                                                      
        # since U and V are scaled by sqrt(rho) * r

        # Ts_params has shape (s x Nparams). Here, just integrating over the radius axis
        Ts_params = np.zeros((lens, Nparams))

        # \Omega_j = \Omega_{in} F_{j1} + \Omega_{out} F_{j2}
        Ts_params[:,0] = simps(Tsr[:,:rcz_ind], axis=1, x=GVAR.r[:rcz_ind]) 
        Ts_params[:,1] = simps(Tsr[:,rcz_ind:], axis=1, x=GVAR.r[rcz_ind:])

        # multiplying and adding the Omegasr_step_params of shape (s x Nparams)
        Ts_params_Omegas_params = Ts_params * Omegas_step_params    # shape (s x Nparams)

        Ts_Omegas = np.sum(Ts_params_Omegas_params, axis=1)  # summing over Nparams axis

        prod_gammas = FN.gamma(ell) * FN.gamma(ell)  # * FN.gamma(s_arr), not using for Omega

        # the factor of 2\omeganl has been taken off as compared to LR92 since we want \delta\omega
        Cvec = FN.minus1pow_vec(m) * 4*np.pi * (wigvals @ (prod_gammas * Ts_Omegas))

        return Cvec

# {{{ def compute_splitting_from_step_params_for_Delta_Omega():
def compute_splitting_from_step_params_for_Delta_Omega(GVAR, Omegas_step_params, mult, rcz_ind):
        """Function to compute the frequency splittings
        under isolated multiplet approximation from the step params.
        - GVAR: Dictionary of global parameters
        - mult: Isolated multiplet whose splitting is to be calculated
        """
        n, ell = mult[0][0], mult[0][1]
        m = np.arange(-ell, ell+1)

        # calculating the s_arr from w_s(r) shape
        lens = Omegas_step_params.shape[0]
        Nparams = Omegas_step_params.shape[1]

        s_arr = np.arange(1, (2*lens - 1) + 1, 2)

        # mult_idx = FN.nl_idx(GVAR, n, ell)
        # omeganl = GVAR.omega_list[mult_idx]

        wigvals = np.zeros((2*ell+1, len(s_arr)))
        for i in range(len(s_arr)):
            wigvals[:, i] = FN.w3j_vecm(ell, s_arr[i], ell, -m, 0*m, m)

        Tsr = compute_Tsr(GVAR, mult, mult, s_arr)
        
        # Loading the synthetic flow profile
        # -1 factor from definition of toroidal field                                                                                                                                   

        # wsr[0, :] *= 0.0 # setting w1 = 0                                                                                                                                            
 
        # wsr[1, :] *= 0.0 # setting w3 = 0                                                                                                                                              
        # wsr[2, :] *= 0.0 # setting w5 = 0                                                                                                                                              
        
        # integrating over radial grid
        # integrand = Tsr * wsr * (self.sup.gvar.rho * self.sup.gvar.r**2)[NAX, :]                                                                                                      
        # since U and V are scaled by sqrt(rho) * r

        # Ts_params has shape (s x Nparams). Here, just integrating over the radius axis
        Ts_params = np.zeros((lens, Nparams))

        # the \Omega_j = \Omega_{out} (F_{j1} + F_{j2}) + \Delta \Omega F_{j1)
        Ts_params[:,0] = simps(Tsr[:,:], axis=1, x=GVAR.r[:]) 
        Ts_params[:,1] = simps(Tsr[:,:rcz_ind], axis=1, x=GVAR.r[:rcz_ind])

        # multiplying and adding the Omegasr_step_params of shape (s x Nparams)
        Ts_params_Omegas_params = Ts_params * Omegas_step_params    # shape (s x Nparams)

        Ts_Omegas = np.sum(Ts_params_Omegas_params, axis=1)  # summing over Nparams axis

        prod_gammas = FN.gamma(ell) * FN.gamma(ell)  # * FN.gamma(s_arr), not using for Omega

        # the factor of 2\omeganl has been taken off as compared to LR92 since we want \delta\omega
        Cvec = FN.minus1pow_vec(m) * 4*np.pi * (wigvals @ (prod_gammas * Ts_Omegas))

        return Cvec
# }}} def compute_splitting_from_step_params_for_Delta_Omega

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
