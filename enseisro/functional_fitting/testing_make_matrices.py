import numpy as np
from enseisro import globalvars
import enseisro.misc_functions as FN
from enseisro.synthetics import create_synth_modes as make_modes 
import matplotlib.pyplot as plt
from enseisro.functional_fitting import make_inversion_matrices as make_inv_mat
from enseisro.synthetics import create_synthetic_DR as create_synth_DR
from enseisro.synthetics import w_omega_functions as w_om_func
from enseisro import get_kernels as get_kerns 
from enseisro.functional_fitting import a_solver as a_solver
from enseisro import forward_functions_Omega as forfunc_Om
import matplotlib.pyplot as plt

ARGS = FN.create_argparser()
GVAR = globalvars.globalVars(ARGS)

#### testing the construction of the data matrix d ##################

# defining the multiplets
mults = np.array([[2,10], [2,12], [3,14]], dtype='int')

# getting the modes for which we want splitting values
modes = make_modes.make_modes(mults)
print(modes)
print(modes[0,0],modes[1,0])
# creating the uncertainty of splitting vector                                                                                                                                          
sigma_arr = np.ones(modes.shape[1])

smax = 1
s_arr = np.arange(1,smax+1,2)
# extracting the solar DR profile in terms of wsr                                                                                                                                       
wsr = create_synth_DR.get_solar_DR(GVAR, smax=smax)
# converting this wsr to Omegasr                                                                                                                                                     
Omegasr = w_om_func.w_2_omega(GVAR, wsr)

# making it  in the (Nstars x s x r) shape
Omegasr = np.reshape(Omegasr, (1, len(s_arr), len(GVAR.r))) 

# getting the data matrix
d = make_inv_mat.make_d_synth(GVAR, modes, sigma_arr, Omega_synth = Omegasr[0])  # using a single star for now                                                                          

print('Here',Omegasr.shape)

# checking if d is correct
domega = np.array([])
for mult in mults:
    domega = np.append(domega, forfunc_Om.compute_splitting(GVAR, Omegasr[0], np.array([mult])))

cumulative_modes = np.arange(0, 2*np.sum(mults[:,1])+len(mults))
all_ms = np.arange(-10,11)
all_ms = np.append(all_ms, np.arange(-12,13))
all_ms = np.append(all_ms, np.arange(-14,15))
print(domega.shape, cumulative_modes.shape)
plt.plot(cumulative_modes, domega*GVAR.OM*1e9,'.k')
plt.axhline(0.0)
shift = 0
mult_marr = np.array([-2,-1,1,2])
start_ind, end_ind = 0, 4
for i in range(len(mults)):
    print(cumulative_modes[mult_marr + shift + mults[i,1]])
    print(all_ms[mult_marr + shift + mults[i,1]])
    plt.plot(cumulative_modes[mult_marr + shift + mults[i,1]], GVAR.OM*1e9*d[start_ind:end_ind],'xr')
    shift += (2*mults[i,1]+1)
    start_ind += 4
    end_ind += 4
plt.grid(True)
plt.savefig('../../plotter/check_d_matrix.pdf')
                   
########## testing the construction of the A matrix ##################

# obtaining the kernel for a certain multiplet
K = get_kerns.compute_kernel(GVAR, np.array([[2,10]]), 0.7, s_arr)
# print(K)
# obtaining the A matrix

A = make_inv_mat.make_A(GVAR, modes, sigma_arr)
# print(A)

# finding the rcz index                           
rcz = 0.7
rcz_ind = np.argmin(np.abs(GVAR.r - rcz))
rcz_ind_arr = np.array([rcz_ind])   # contains only one star 

# calculating the step function equivalent of this Omegasr                                                                                                                              
step_param_arr = create_synth_DR.get_solar_stepfn_params(Omegasr, rcz_ind_arr)   # shape (Nstars x Nparams x s)                                                                         
Omegasr_step = create_synth_DR.params_to_step(GVAR, step_param_arr, rcz_ind_arr)

print(Omegasr_step.shape)

# solving for a in A . a = d
a = a_solver.use_numpy_inv(GVAR, modes, sigma_arr, smax, use_synth=True, Omegasr = Omegasr_step[0])
# print(Omegasr_step * GVAR.OM * 1e9)
print(a * GVAR.OM * 1e9)

