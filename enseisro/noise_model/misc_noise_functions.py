# import jax.numpy as np
import numpy as np

# this file contains miscellaneous functions to construct the Noise model.
# the nomenclature is adopted from Thorsten Stahn's PhD thesis.

# {{{ def make_N_nu():
def make_N_nu(nu_arr, tau_arr, A_arr):
    """This function returns $N(\nu)$ from a given frequency array $\nu$.

    Parameters
    ----------
    nu_arr : numpy.ndarray    
        Frequency array.          
    tau_arr : numpy.ndarray
        Time-scale array.
    A_arr : numpy.ndarray
        Amplitude array.
    """

# }}} def make_N_nu()
