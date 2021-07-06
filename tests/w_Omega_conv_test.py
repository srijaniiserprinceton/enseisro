# import jax.numpy as np
import numpy as np
from enseisro import forward_functions_w as forfunc_w
from enseisro import forward_functions_Omega as forfunc_Om
from enseisro.synthetics import create_synthetic_DR as create_synth_DR
from enseisro.synthetics import w_omega_functions as w_om_func
from enseisro import globalvars
import enseisro.misc_functions as FN

ARGS = FN.create_argparser()
GVAR = globalvars.globalVars(ARGS)

def test_forward_calc():
    """This function tests whether for an equivalent wsr or Omegasr profile,
    the frequency splittings calculated from two different forward problems
    (where essentially the kernels vary slightly) are the same or not."
    """
    # choosing a certain multiplet
    mult = np.array([[2,10]])
    
    # extracting the solar DR profile in terms of wsr
    wsr = create_synth_DR.get_solar_DR(GVAR, smax=1)

    # converting this wsr to Omegasr
    Omegasr = w_om_func.w_2_omega(GVAR, wsr)

    # getting the respective frequency splittings
    domega_w = forfunc_w.compute_splitting(GVAR, wsr, mult)
    domega_Om = forfunc_Om.compute_splitting_from_function(GVAR, Omegasr, mult)

    np.testing.assert_array_almost_equal(domega_w, domega_Om)
