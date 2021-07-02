# import jax.numpy as np
import numpy as np
from enseisro.functional_fitting import make_inversion_matrices as make_mat

def use_numpy_inv(GVAR, modes, sigma_arr, smax, use_synth=True, Omegasr=np.array([])):
    """This is to calculate a in the equation A . a = d, using the
    Numpy solver numpy.linalg.inv()"""
    
    A = make_mat.make_A(GVAR, modes, sigma_arr, smax=smax)
    d = make_mat.make_d_synth(GVAR, modes, sigma_arr, Omega_synth=Omegasr)
    
    AT = A.T

    ATA = AT @ A
    
    a = np.linalg.inv(ATA) @ (AT @ d)

    return a
