# import jax.numpy as np
import numpy as np
import scipy as sp
from enseisro.functional_fitting import make_inversion_matrices as make_mat

def use_numpy_inv(GVAR, modes, sigma_arr, smax, use_synth=True, Omegasr=np.array([])):
    """This is to calculate a in the equation A . a = d, using the
    Numpy solver numpy.linalg.inv()"""
    
    A = make_mat.make_A(GVAR, modes, sigma_arr, smax=smax)
    d = make_mat.make_d_synth_from_function(GVAR, modes, sigma_arr, Omegasr)

    
    AT = A.T

    ATA = AT @ A
    
    inv_ATA = sp.linalg.inv(ATA) 

    # computing solution
    a = inv_ATA @ (AT @ d)

    # computing the model covariance matrix.
    # C_M = (A^T A)^{-1} according to Sambridge
    # Lecture notes. Lecture 2, Slide 23.
    C_M = inv_ATA

    return a, C_M
