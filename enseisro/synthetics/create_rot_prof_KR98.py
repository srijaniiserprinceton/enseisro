# import jax.numpy as np
import numpy as np

day2sec = 24 * 3600

# {{{ def get_DeltaOmega_from_Prot():
def get_DeltaOmega_from_Prot(Prot, star_type):
    """Returns the radial differential rotational gradient
    Delta Omega from the observed surface rotational time-period.
    
    Parameters
    ----------
    Prot : array_like, float
        Latitudinally averaged surface rotation period in days.
    """

    # converting from days to seconds
    Prot_sec = Prot * day2sec

    # Omega_surface in nHz
    Omega = (1 / Prot_sec) * 1e9
    
    if(star_type == 'G2'): Delta_Omega = make_G2_DeltaOmega(Omega)

    # in nHz. Same shape as Prot
    return Delta_Omega, Omega

# }}} def get_DeltaOmega_from_Prot()


# {{{ def make_G2_DeltaOmega():
def make_G2_DeltaOmega(Omega):
    """Constructs Delta Omega according to power law
    relation given in Kitchatinov & Rudiger 1998.
    
    Parameters
    ----------
    Omega : array_like, float
        Latitudinally averaged rotation rate in nHz.
    """
    
    plaw_exp_arr = np.array([1, 1.56])     
    Prot = np.array([1, 28])   # in days
    # values taken from Fig 1 of KR98
    Delta_Omega_by_Omega = np.array([5e-4, 0.015])
    Prot_sec = Prot * day2sec  # in seconds
    Omega_plaw = (1 / Prot_sec) * 1e9  # in nHz
    
    print('Omega power law in nHz:\n', Omega_plaw)
    # fitting straight line
    # poly_fit = np.polyfit(Omega_plaw, Delta_Omega_by_Omega, 1)
    plaw_exp_fit = np.polyfit(Omega_plaw, plaw_exp_arr, 1)

    # getting the powerlaw exponent for a custom Omega
    exp_for_Omega = np.poly1d(plaw_exp_fit)(Omega)

    # solar rotation factor 
    # const_prop = 7 / 441**(1 - exp_for_Omega)

    # getting the Delta Omega for a custom Omega
    # using \Delta\Omega / \Omega ~ \Omega^{-n'} in KR1998
    # Delta_Omega_by_Omega = np.poly1d(poly_fit)(Omega) 
    # Delta_Omega = Delta_Omega_by_Omega * Omega
    Delta_Omega = Omega**(1 - exp_for_Omega)

    # in nHz
    return Delta_Omega
    
