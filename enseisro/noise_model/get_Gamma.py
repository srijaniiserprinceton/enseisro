# import jax.numpy as np
import numpy as np

# {{{ def get_Gamma():
def get_Gamma(n_arr, ell_arr):
    """Returns the Gamma vector for a given
    set of multiplets.
    
    Parameters
    ----------
    n_arr : int, array_like
        The array of radial orders for all modes.
    ell_arr : int, array_like
        The array of angular degrees for all modes
    """
    # the raw array containing the linewidths from Stahn's thesis,
    # Table 2.4. This is for radial orders n = 15 to 28. In muHz
    Gamma_0 = np.array([0.95, 1.06, 1.09, 1.04, 0.97, 0.99, 1.07, 
                        1.35, 1.85, 2.64, 3.84, 5.42, 7.39, 9.96])

    # the radial orders corresponding to the above. n = 15 to 28
    n_arr_for_Gamma_0 = np.arange(15,29)

    # constructing Gamma array to be returned
    Nmodes = len(n_arr)
    Gamma_arr = np.zeros(Nmodes)
    
    for i in range(Nmodes):
        n, ell = n_arr[i], ell_arr[i]
        
        # if ell is 0 or 1 then Gamma_0 value is stored for that n
        if(ell < 2):
            n_ind = np.argmin(np.abs(n - n_arr_for_Gamma_0))
            Gamma_arr[i] = Gamma_0[n_ind]

        # if not then, Gamma_0 value for the (n+1)th mode is stored.
        # this puts an upper bound on the highest radial order we can use for inversions
        # when using ell = 2 angular degrees
        else:
            n_ind = np.argmin(np.abs(n - n_arr_for_Gamma_0))
            Gamma_arr[i] = Gamma_0[n_ind+1]

    # has length Nmodes
    return Gamma_arr
            

# }}} def get_Gamma()
