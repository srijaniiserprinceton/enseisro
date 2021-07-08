import numpy as np
from enseisro import forward_functions_Omega as forfunc_Om
from enseisro import globalvars
import enseisro.misc_functions as FN
from enseisro.synthetics import create_synthetic_DR as create_synth_DR
from enseisro.synthetics import w_omega_functions as w_om_func

# ARGS = FN.create_argparser()
# GVAR = globalvars.globalVars(ARGS)

GVAR = globalvars.globalVars()

# {{{ def test_smooth_and_step_consistency():
def test_smooth_and_step_consistency():
    # defining the multiplets                                                                                                                    
    mults = np.array([[2,10]], dtype='int')

    smax = 1
    s_arr = np.arange(1,smax+1,2)
    # extracting the solar DR profile in terms of wsr  
    wsr = create_synth_DR.get_solar_DR(GVAR, smax=smax)
    # converting this wsr to Omegasr                                                                                                                                                    \
    Omegasr = w_om_func.w_2_omega(GVAR, wsr)

    # making it  in the (Nstars x s x r) shape                                                                                                                                               
    Omegasr = np.reshape(Omegasr, (1, len(s_arr), len(GVAR.r)))
    
    # finding the rcz index                                                                                                                                                                  
    rcz = 0.7
    rcz_ind = np.argmin(np.abs(GVAR.r - rcz))
    rcz_ind_arr = np.array([rcz_ind])   # contains only one star                                                                                                                             
    
    # calculating the step function equivalent of this Omegasr                                                                                                                              
    step_param_arr = create_synth_DR.get_solar_stepfn_params(Omegasr, rcz_ind_arr)   # shape (Nstars x s x  Nparams)                                                                        \

    # calculating the smooth step function over radius
    Omegasr_step = create_synth_DR.params_to_step(GVAR, step_param_arr, rcz_ind_arr)

    print(Omegasr_step.shape)

    # getting the rotational splittings from the step function params
    domega_step_params = forfunc_Om.compute_splitting_from_step_params(GVAR, step_param_arr[0], mults, rcz_ind)
    
    # getting the rotational splittings from the total function in radius
    domega_step_function = forfunc_Om.compute_splitting_from_function(GVAR, Omegasr_step[0], mults)
    
    np.testing.assert_array_almost_equal(domega_step_params, domega_step_function)
# }}} def test_smooth_and_step_consistency()

# test_smooth_and_step_consistency()
