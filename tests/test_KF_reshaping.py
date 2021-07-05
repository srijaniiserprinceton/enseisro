# import jax.numpy as np
import numpy as np
from enseisro import misc_functions as FN

def test_reshape_KF():
    """This function checkes reshaping of the KF matrix of shape (modes x Nparams x s)
    to the shape (modes x (Nparams*lens)). We are creating this function mainly to 
    be sure that this is what is happening all along. The reshaping is important and
    subtle since there are two style 'F' and 'C' style. We are going to use the 'F' style.
    """
    a = np.array([[1,2,3],[4,5,6]])
    a = a.T
    # creating a sample matrix of dimension (Nmodes x Nparams x s) 
    KF = np.zeros((3,2,3))

    # populating the KF matrix for easily checking 
    KF[:,:,0] = a
    KF[:,:,1] = 2*a
    KF[:,:,2] = 3*a
    
    # now reshaping the way we want it to happen
    # KF = np.reshape(KF,(KF.shape[0],-1),'F')
    KF = FN.reshape_KF(KF)

    # this should match with the following matrix
    KF_correct = np.array([[ 1.,  4.,  2.,  8.,  3., 12.],
                           [ 2.,  5.,  4., 10.,  6., 15.],
                           [ 3.,  6.,  6., 12.,  9., 18.]]) 

    # checking the KF reshaping
    np.testing.assert_array_almost_equal(KF,KF_correct)

    

