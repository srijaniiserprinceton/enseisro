# import jax.numpy as np
import numpy as np

NAX = np.newaxis

WFNAME = 'w_s/w.dat'

# {{{ get_solar_DR():
def get_solar_DR(GVAR, smax=3):
    '''Function to load and return the Solar differential rotation
    in the w_s(r) form for s=1,3 by default
    '''
    wsr = np.loadtxt(f'{GVAR.datadir}/{WFNAME}')\
              [:, GVAR.rmin_idx:GVAR.rmax_idx] * (-1.0)

    # slicing the w_s(r) profile according to smax
    wsr = wsr[:(smax-1)//2 + 1]

    return wsr
# }}} get_solar_DR()

def make_DR_copies_random(GVAR, wsr, N=10, p=1):
    '''This function returns multiple copies of
    an input DR profile input in the w_s(r) format'''
    slen, rlen = np.shape(wsr)
    wsr_ens = np.zeros((slen,rlen,N))
    # initializing with the same profile first
    # will add the randomness next
    # Shape (s x r x N)
    wsr_ens += wsr[:,:,NAX] 
    
    # random array to be added. Shape (1 X N)
    # generating random numbers between [-1,1]
    rand_arr = 0.5 * (np.random.rand(N) - 0.5)

    # introducing upto a p-percent difference from 
    # original w_s(r)
    wsr_ens *= (1 + rand_arr * p * 0.01)[NAX, NAX, :]

    return wsr_ens
    

    
