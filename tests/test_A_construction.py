import numpy as np
from enseisro import globalvars
import enseisro.misc_functions as FN
from enseisro.synthetics import create_synth_modes as make_modes
import matplotlib.pyplot as plt
from enseisro.functional_fitting import make_inversion_matrices as make_inv_mat
from enseisro.synthetics import create_synthetic_DR as create_synth_DR
from enseisro.synthetics import w_omega_functions as w_om_func
from enseisro import get_kernels as get_kerns
from enseisro import forward_functions_Omega as forfunc_Om
import matplotlib.pyplot as plt

ARGS = FN.create_argparser()
GVAR = globalvars.globalVars(ARGS)

NAX = np.newaxis

def test_K_construction(plot=False):
    # defining the multiplets                                                                                                                                                                
    mults = np.array([[2,10], [3,12], [4,14]], dtype='int')
    # getting the modes for which we want splitting values                                                                                                                                   
    modes = make_modes.make_modes(mults)
    # the smax and s array
    smax = 1
    s_arr = np.arange(1,smax+1,2)
    
    # finding the rcz index                                                                                                                                                             
    rcz = 0.7
    rcz_ind = np.argmin(np.abs(GVAR.r - rcz))
    rcz_ind_arr = np.array([rcz_ind])   # contains only one star 

    # the uncertainty vector
    sigma_arr = np.ones(modes.shape[1])
    
    # getting constructing the A matrix
    A = make_inv_mat.make_A(GVAR, modes, sigma_arr, rcz=rcz, smax=smax)    # shape (Nmodes x Nparams)

    # constructing Omegasr to calculate frequency splittings in the next step
    # extracting the solar DR profile in terms of wsr                                                                                                                                   
    wsr = create_synth_DR.get_solar_DR(GVAR, smax=smax)
    # converting this wsr to Omegasr. Shape (s x r)                                                                                                                                     
    Omegasr = w_om_func.w_2_omega(GVAR, wsr)
    # making it  in the (Nstars x s x r) shape                                                                                                                                         
    Omegasr = np.reshape(Omegasr, (1, len(s_arr), len(GVAR.r)))
     
    
    # calculating the step function equivalent of this Omegasr                                                                                                                          
    step_param_arr = create_synth_DR.get_solar_stepfn_params(Omegasr, rcz_ind_arr)   # shape (Nstars x s x Nparams)                                                                     
    Omegasr_step = create_synth_DR.params_to_step(GVAR, step_param_arr, rcz_ind_arr) # shape (Nstars x s x r)


    # computing the frequency splittings using the Omegasr profile 
    domega_A_test = np.sum(A * step_param_arr[NAX,0,0,:], axis=1) 

    # getting splitting from debug function
    domega_splitting = np.array([])
    for mult in mults:
        domega_splitting = np.append(domega_splitting, forfunc_Om.compute_splitting_from_function(GVAR, Omegasr_step[0], np.array([mult])))

    # finding the values that is present in our domega_A_test
    mult_marr = np.array([-2,-1,1,2])
    domega_splitting_in_A_test = np.array([])

    # extracting the domegas present in d                                                                                                                                               
    shift = 0
    for i in range(len(mults)):
        domega_splitting_in_A_test = np.append(domega_splitting_in_A_test, domega_splitting[mult_marr + shift + mults[i,1]])
        shift += (2*mults[i,1]+1)

    np.testing.assert_array_almost_equal(domega_A_test, domega_splitting_in_A_test)

    # cumulative mode label for plottinh                                                                                                                                                             
    cumulative_modes = np.arange(0, 2*np.sum(mults[:,1])+len(mults))

    # for visual confirmation                                                                                                                                                           
    if(plot):
        plt.plot(cumulative_modes, domega_splitting*GVAR.OM*1e9,'.k')
        plt.axhline(0.0)
        shift = 0
        start_ind, end_ind = 0, 4
        for i in range(len(mults)):
            plt.plot(cumulative_modes[mult_marr + shift + mults[i,1]], GVAR.OM*1e9*domega_A_test[start_ind:end_ind],'xr')
            shift += (2*mults[i,1]+1)
            start_ind += 4
            end_ind += 4
        plt.grid(True)
        plt.savefig('../plotter/check_A_matrix.pdf')

# visual confirmation
# test_K_construction(plot=True)
