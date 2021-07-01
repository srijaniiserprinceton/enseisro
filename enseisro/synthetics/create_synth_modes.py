# import jax.numpy as np
import numpy as np

# def make_A():
#    """This function is used to create the matrix in
#    A . a = d for our linear inverse problem
#    """


def make_modes(mults):
    """This function is used to build specific modes for synthetic tests.
    """
    nlm_array = np.array([])  # array containing the n,l,m values of a mode
    
    # counter to be executed only once
    is_first = True
    
    for i, mult in enumerate(mults):
        # we restrict ourselves to m = \pm 1,\pm 2 only
        m_arr = np.array([-2,-1,1,2])
        for m in m_arr:
            mode = np.array([mult[0], mult[1], m])
            # it is okay to hardcode the shape since it won't change
            mode = np.reshape(mode,(3,1))
            
            # ensure the shape is readjusted correctly the first time
            if(is_first):
                nlm_array = np.append(nlm_array, mode)
                # reshaping to allow correct appending. It is okay
                # to hardcode the shape to (3,1) since that won't change
                nlm_array = np.reshape(nlm_array, (3,1))
                # turning off counter since we have the correct shape of nlm_array now
                is_first = False
                
            # from the second time onwards
            else:
                nlm_array = np.append(nlm_array, mode, axis=1)
            print(nlm_array.shape)

    print(nlm_array)
