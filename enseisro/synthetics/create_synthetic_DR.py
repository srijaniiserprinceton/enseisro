# import jax.numpy as np
import numpy as np

NAX = np.newaxis

WFNAME = 'w_s/w.dat'

# {{{ def get_solar_DR():
def get_solar_DR(GVAR, smax=3):
    '''Function to load and return the Solar differential rotation
    in the w_s(r) form for s=1,3 by default
    '''
    wsr = np.loadtxt(f'{GVAR.datadir}/{WFNAME}')\
              [:, GVAR.rmin_idx:GVAR.rmax_idx] * (-1.0)

    # slicing the w_s(r) profile according to smax
    wsr = wsr[:(smax-1)//2 + 1]

    return wsr
# }}} def get_solar_DR()

# {{{ def make_DR_copies_random():
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
# }}} def make_DR_copies_random()

# {{{ def make_solar_stepfn():    
def get_solar_stepfn_params(Omegasr, rcz_ind_arr, Nparams=2):
    """This function receives the Omega_s(r) for a number of stars
    and returns the corresponding step functions by averaging the 
    interior and exterior zones (about the rcz).
    """
    # Omegasr needs to be passed in the shape (Nstars x Nparams x s)

    # finding the number of stars passed
    Nstars, lens = Omegasr.shape[0], Omegasr.shape[1]
    
    # creating the array to store \Omega_{out} and \Omega_{out} + \Delta \Omega
    step_param_arr = np.zeros((Nstars, lens, Nparams))   # shape (Nstars x s x Nparams)

    for star_ind in range(Nstars):
        step_param_arr[star_ind, :, 0] = np.mean(Omegasr[star_ind,:,:rcz_ind_arr[star_ind]], axis=1)
        step_param_arr[star_ind, :, 1] = np.mean(Omegasr[star_ind,:,rcz_ind_arr[star_ind]:], axis=1)
                              
    return step_param_arr
# }}} def_make_solar_stepfn_params()

def params_to_step(GVAR, step_param_arr, rcz_ind_arr):
    Nstars, lens = step_param_arr.shape[0], step_param_arr.shape[1]
    Omegasr_step = np.zeros((Nstars,lens,len(GVAR.r)))

    for s_ind in range(lens):
        for star_ind in range(Nstars):
            rcz_ind = rcz_ind_arr[star_ind]
            Omegasr_step[star_ind,s_ind,:rcz_ind] = step_param_arr[star_ind,s_ind,0]
            Omegasr_step[star_ind,s_ind,rcz_ind:] = step_param_arr[star_ind,s_ind,1]

    # returning shape (Nstars x s x r)
    return Omegasr_step
