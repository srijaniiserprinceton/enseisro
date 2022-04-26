# import jax.numpy as np
import numpy as np

# {{{ def convert_Hnlm():
def convert_Hnlm(Hnlm_sun, rel_gravity_arr):
    """Converts Hnlm_sun to Hnlm_stars using the relation
    Hnlm_stars/Hnlm_sun = g_sun^2/g_stars^2 from Chaplin et al 09.
    
    Parameters
    ----------
    Hnlm_sun : array_like, float
        Hnlm of the sun for the given set of modes.
    rel_gravity_arr : array_like, float
        Relative surface gravity of stars with respect to Sun.
    """

    Hnlm_stars = Hnlm_sun * 1./(rel_gravity_arr**2)

    return Hnlm_stars

# }}} def convert_Hnlm()


# {{{ def convert_Gamma():
def convert_Gamma(Gamma_sun, rel_Teff_arr):
    """Converts Gamma_sun to Gamma_stars using the relation
    Gamma_stars/Gamma_sun = Teff_stars^4/Teff_sun^4 from Chaplin et al 09.
    
    Parameters
    ----------
    Gamma_sun : array_like, float
        Linewidths of the sun for the given set of modes.
    rel_Teff_arr : array_like, float
        Relative T_eff of stars with respect to Sun.
    """

    Gamma_stars = Gamma_sun * (rel_Teff_arr**4)

    return Gamma_stars

# }}} def convert_Gamma()


# {{{ def convert_B_nu():
def convert_B_nu(B_nu_sun, rel_numax_arr):
    """Converts B_nu_sun to B_nu_stars using the relation
    B_nu_stars/B_nu_sun = numax_stars/numax_sun.
    
    Parameters
    ----------
    B_nu_sun : array_like, float
        B_nu of the sun for the given set of modes.
    rel_numax_arr : array_like, float
        Relative \nu_max of stars with respect to Sun.
    """

    B_nu_stars = B_nu_sun * rel_numax_arr

    return B_nu_stars

# }}} def convert_B_nu()
