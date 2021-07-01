# import jax.numpy as np
import numpy as np

# def make_A():
#    """This function is used to create the matrix in
#    A . a = d for our linear inverse problem
#    """


def make_d(modes, use_synth=True, Omega_synth=np.array([])):
    """This function is used to build the data vector d  of frequency splittings
    in the forward problem A . a = d
    """
    
    
