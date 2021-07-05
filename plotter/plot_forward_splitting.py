import numpy as np
from enseisro import globalvars
import enseisro.misc_functions as FN
import enseisro.forward_functions_w as forfunc_w
import enseisro.forward_functions_Omega as forfunc_Om
import matplotlib.pyplot as plt
from enseisro.synthetics import create_synthetic_DR as create_syn_DR
from enseisro.synthetics import w_omega_functions as w_Om_func

ARGS = FN.create_argparser()
GVAR = globalvars.globalVars(ARGS)

def plot_solar_splitting(wsr, n=2, ell=10):
    mult = np.array([[n,ell]])

    domega_m = forfunc_w.compute_splitting(GVAR, wsr, mult)
    # converting to nHz
    domega_m *= GVAR.OM * 1e9

    # doing the same for Omega instead of wsr
    Omega_sr = w_Om_func.w_2_omega(GVAR, wsr)
    domega_m_Om = forfunc_Om.compute_splitting(GVAR, Omega_sr, mult)
    domega_m_Om *= GVAR.OM * 1e9

    m_arr = np.arange(-mult[0][1], mult[0][1]+1)

    plt.figure()
    
    plt.plot(m_arr, domega_m ,'ok')
    plt.plot(m_arr, (domega_m - domega_m_Om), '-r')
    plt.xlabel('$m$')
    plt.ylabel('$\delta \omega^{%i,%i}(m)$'%(n,ell))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('freq_splitting_sun.pdf')
    plt.close()

def plot_solarlike_ens_splitting(wsr_ens, n=2, ell=10, sarr=np.array([1,3,5])):
    mult = np.array([[n,ell]])
    
    # total number of stars in the ensemble
    lenN = wsr_ens.shape[-1]

    # creating the m array
    m_arr = np.arange(-mult[0][1], mult[0][1]+1)
    
    plt.figure()

    for i in range(lenN):
        domega_m = forfunc_w.compute_splitting(GVAR, wsr_ens[:,:,i], mult)
        
        # converting to nHz                                                                      
        domega_m *= GVAR.OM * 1e9
        
        # plotting the freq splitting for a multiplet as a function of m 
        plt.plot(m_arr, domega_m)

    plt.xlabel('$m$')
    plt.ylabel('$\delta \omega^{%i,%i}(m)$'%(n,ell))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('freq_splitting_ens.pdf')
    plt.close()

# getting the solar DR
wsr = create_syn_DR.get_solar_DR(GVAR, smax=5)
# plotting the splitting for the Sun
plot_solar_splitting(wsr)

# getting the ensemble of sun-like DR
wsr_ens = create_syn_DR.make_DR_copies_random(GVAR, wsr, N=10, p=100)
# plotting the splitting for Sunlike DR
plot_solarlike_ens_splitting(wsr_ens)
