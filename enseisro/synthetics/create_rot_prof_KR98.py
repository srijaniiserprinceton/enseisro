# import jax.numpy as np
import numpy as np

day2sec = 24 * 3600

# {{{ def get_DeltaOmega_from_Prot():
def get_DeltaOmega_from_Prot(Prot_days, spectral_type):
    """Returns the radial differential rotational gradient
    Delta Omega from the observed surface rotational time-period.
    
    Parameters
    ----------
    Prot_days : array_like, float
        Latitudinally averaged surface rotation period in days.
    spectral_type : string
        The spectral class of the stars in the ensemble.
    """
    
    # creating Prot in seconds
    Prot_sec = Prot_days * day2sec 

    # creating Omega in nHz
    Omega = (1.0 / Prot_sec) * 1e9

    # getting the Delta Omega in nHz
    if(spectral_type == 'G2'): DOmega_by_Omega = make_G2_DeltaOmega(Prot_days)

    # scaling since KR98 has underestimated
    DOmega_by_Omega *= 5

    # in nHz. Same shape as Prot
    return DOmega_by_Omega, Omega

# }}} def get_DeltaOmega_from_Prot()


# {{{ def make_G2_DeltaOmega():
def make_G2_DeltaOmega(Prot_days):
    """Constructs Delta Omega according to power law
    relation given in Kitchatinov & Rudiger 1998.
    
    Parameters
    ----------
    Prot_days : array_like, float
        Latitudinally averaged rotation period in days.
    """
    # the value of n' as in KR98
    plaw_exp = np.array([1.2, 1.02])
    
    # values to make the polynomial
    Prot_days_fit = np.array([1, 28])   # in days

    # values taken from Fig 1 of KR98
    DOmega_by_Omega_fit = np.array([5e-4, 0.015])

    # fitting the slope as a straight line
    plaw_exp_fit = np.polyfit(np.log10(Prot_days_fit), plaw_exp, 1)

    exp_from_Prot = np.poly1d(plaw_exp_fit)(np.log10(Prot_days))

    
    '''
    # fitting straight line
    plaw_fit_DOmega = np.polyfit(np.log10(Prot_days_fit),\
                    np.log10(DOmega_by_Omega_fit), 1)

    # getting the powerlaw exponent for a custom Omega
    poly_fit = np.poly1d(plaw_fit_DOmega)

    # interpolating
    log10_DOmega_by_Omega = np.poly1d(poly_fit)(np.log10(Prot_days)) 
    
    DOmega_by_Omega = 10**log10_DOmega_by_Omega
    '''
    # y = m * x + c
    log10_DOmega_by_Omega = exp_from_Prot * np.log10(Prot_days) +\
                            np.log10(DOmega_by_Omega_fit[0])

    DOmega_by_Omega = 10**log10_DOmega_by_Omega

    # in nHz
    return DOmega_by_Omega
    

# {{{ def make_step_param_arr():
def make_step_param_arr(Nstars, Prot, spectral_type='G2'):
    """Returns the step_param_arr array containing the 
    Omega_{out} and Delta Omega values for each star in the 
    ensemble.
    """
    # number of unique stars who rotation profile the user wants
    N_unique_stars = len(Prot)
    # checking if len(Prot) is 1. If so, then the user needs 
    # to build Nstars identical stars' rotation profile.
    if(N_unique_stars):
        # getting the step params in nHz
        DOmega_by_Omega, Omega = get_DeltaOmega_from_Prot(Prot, spectral_type)
        DOmega = DOmega_by_Omega * Omega

        # creating the step_param_arr of shape (Nstars x s  x Nparams)
        # for now, we are hardcoding s to s = 1 only.
        step_param_arr = np.zeros((Nstars, 1, 2))
        # filling in Omega_out and Delta Omega
        step_param_arr[:,0,0] += Omega
        step_param_arr[:,0,1] += DOmega

        return step_param_arr
    # else we return the same number of stars are len(Prot) since
    # the user has sent in different Prot for different stars in that
    # ensemble
    else:
         # getting the step params in nHz
        DOmega_by_Omega, Omega = get_DeltaOmega_from_Prot(Prot, spectral_type)
        DOmega = DOmega_by_Omega * Omega

        # creating the step_param_arr of shape (Nstars x s  x Nparams)
        # for now, we are hardcoding s to s = 1 only.
        step_param_arr = np.zeros((Nstars, 1, 2))
        # filling in Omega_out and Delta Omega
        step_param_arr[:,0,0] += Omega
        step_param_arr[:,0,1] += DOmega

        return step_param_arr

# }}} def make_step_param_arr()
