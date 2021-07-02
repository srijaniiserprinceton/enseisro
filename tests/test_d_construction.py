import numpy as np
import argparse
from enseisro import globalvars
import enseisro.misc_functions as FN
from enseisro.synthetics import create_synth_modes as make_modes
import matplotlib.pyplot as plt
from enseisro.functional_fitting import make_inversion_matrices as make_inv_mat
from enseisro.synthetics import create_synthetic_DR as create_synth_DR
from enseisro.synthetics import w_omega_functions as w_om_func
from enseisro import forward_functions_Omega as forfunc_Om
import matplotlib.pyplot as plt

ARGS = FN.create_argparser()
GVAR = globalvars.globalVars(ARGS)

def test_d_construction(plot=False):

    #### testing the construction of the data matrix d ##################                                                                                                                    
    
    # defining the multiplets                                                                                                                                                                
    mults = np.array([[2,10], [2,12], [3,14]], dtype='int')
    # getting the modes for which we want splitting values                                                                                                                                   
    modes = make_modes.make_modes(mults)
    # creating the uncertainty of splitting vector                                                                                                                                          
    sigma_arr = np.ones(modes.shape[1])
    
    
    smax = 1
    s_arr = np.arange(1,smax+1,2)
    # extracting the solar DR profile in terms of wsr                                                                                                                                       
    wsr = create_synth_DR.get_solar_DR(GVAR, smax=smax)
    # converting this wsr to Omegasr                                                                                                                                                         
    Omegasr = w_om_func.w_2_omega(GVAR, wsr)
    # making it  in the (Nstars x s x r) shape                                                                                                                                               
    Omegasr = np.reshape(Omegasr, (1, len(s_arr), len(GVAR.r)))
    
    
    # getting the data matrix                                                                                                                                                                
    d = make_inv_mat.make_d_synth(GVAR, modes, sigma_arr, Omega_synth = Omegasr[0])  # using a single star for now                                                                          
    
    # domega will contain all the splittings for all multiplets                                                                                                                                                  
    domega = np.array([])
    for mult in mults:
        domega = np.append(domega, forfunc_Om.compute_splitting(GVAR, Omegasr[0], np.array([mult])))
        
    # cumulative mode label
    cumulative_modes = np.arange(0, 2*np.sum(mults[:,1])+len(mults))
    
    # the marr for the modes used
    mult_marr = np.array([-2,-1,1,2])
    domega_in_d = np.array([])
    
    # extracting the domegas present in d
    shift = 0
    for i in range(len(mults)):
        domega_in_d = np.append(domega_in_d, domega[mult_marr + shift + mults[i,1]])
        shift += (2*mults[i,1]+1)

    # testing
    np.testing.assert_array_almost_equal(d, domega_in_d)

    # for visual confirmation
    if(plot):
        plt.plot(cumulative_modes, domega*GVAR.OM*1e9,'.k')
        plt.axhline(0.0)
        shift = 0
        start_ind, end_ind = 0, 4
        for i in range(len(mults)):
            plt.plot(cumulative_modes[mult_marr + shift + mults[i,1]], GVAR.OM*1e9*d[start_ind:end_ind],'xr')
            shift += (2*mults[i,1]+1)
            start_ind += 4
            end_ind += 4
        plt.grid(True)
        plt.savefig('../plotter/check_d_matrix.pdf')


# for visual confirmation
# test_d_construction(plot=True)
