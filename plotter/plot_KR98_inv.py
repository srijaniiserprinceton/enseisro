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
from enseisro import printing_functions as print_func
from enseisro.noise_model import get_noise_for_ens as get_noise
from enseisro.synthetics import create_rot_prof_KR98 as create_rot_prof
from enseisro.functional_fitting import run_funcfit_inv as run_funcfit_inv
import sys
# import matplotlib 

font = {#'family' : 'normal',
        #'weight' : 'bold',
        'size'   : 16}

plt.rc('font', **font)

# defining the parameters necessary                                                                                                                                            
# Number of stars                                                                                                                                               
Nstars = 10
# fac = 10
# multiply_star_arr = np.array([1e2, 1e3])
multiply_star_arr = np.array([22085, 12894, 5132]) #, 2344])
multiply_star_arr = multiply_star_arr

# Observed mode info                                                                                                                                                          
nmin, nmax = 16, 23
lmin, lmax = 1, 2

# Max angular degree for Omega_s                                                                                                                                             
smax = 1

# Base of the convection zone for each star                                                                                                                                    
rcz_arr = np.zeros(Nstars) + 0.7

# the max. perentage randomization of rotation profiles                                                                                                                       
p = 0.0

# whether to use the noise model to add synthetic noise                                                                                                                      
# to the frequency splitting data                                                                                                                                   
add_noise = True
use_Delta = True
# plotting the power law
# creating the range of Prot we want to use          
Npoints = 100                                                                                                                           
Prot = np.logspace(np.log10(1), np.log10(35), Npoints)

# Delta Omega / Omega                                                                                                                                                         
DOmega_by_Omega_true = np.logspace(np.log10(5e-4), np.log10(0.02), Nstars)

# getting the Delta Omega and Omega in nHz
DOmega_by_Omega_gen, __ = create_rot_prof.get_DeltaOmega_from_Prot(Prot, 'G2')

fig, ax = plt.subplots(1, figsize=(6,6))
ax.set_xscale('log')
ax.set_yscale('log')
ax.plot(Prot, DOmega_by_Omega_gen, 'k')

# Nbins = 6
# Prot_low_arr = Prot[::Npoints//Nbins]

Prot_low_arr = np.array([1., 10., 20.]) #, 30.]) 
Prot_window = 10.0

for i, Prot_low in enumerate(Prot_low_arr):
    print('Ensemble number: %i'%(i+1))
    # the rotation period of the stars in days                                                                                                                             
    Prot_high = Prot_low + Prot_window
    Prot = np.linspace(Prot_low, Prot_high, Nstars)
    # Prot = np.array([Prot_low])                                                                                                                                                        
    # carrying out the inversion and printing the results and getting errors in nHz                                                                                                     
    Omega_avg, DOmega, err_Omega_avg, err_DOmega, avg_DOmega_by_Omega = run_funcfit_inv.run_ens_inv(Prot, Nstars, nmin, nmax, lmin, lmax,\
                                                          smax, rcz_arr, p, add_noise=add_noise, use_Delta=use_Delta)
    
    
    # final plot params in nHz
    DOmega_by_Omega_avg, err_DOmega_by_Omega_avg = FN.propagate_errors(avg_DOmega_by_Omega, Omega_avg, DOmega, err_Omega_avg, err_DOmega)
    # DOmega_by_Omega_avg, err_DOmega_by_Omega_avg = FN.propagate_errors(avg_DOmega_by_Omega, err_Omega_avg, err_DOmega)

    # plotting the inverted value with error bars
    # for j, multiply_star in enumerate(multiply_star_arr):
    fac = multiply_star_arr[i]    # different numbers for different Prot windows
    ax.errorbar(np.mean(Prot), avg_DOmega_by_Omega, yerr=err_DOmega_by_Omega_avg/np.sqrt(fac/Nstars),\
                                            fmt = 'o', capsize=10, label='%.0e'%(fac))
        
    # if(i==0): ax.legend(loc='upper center', bbox_to_anchor=(0.5,1.1), ncol=len(multiply_star_arr), prop={'size' : 12})
    
    print('Prot_low', Prot_low)

    # resetting color order 
    # plt.gca().set_prop_cycle(None)
    # print(err_DOmega_by_Omega_avg/np.sqrt(fac))

ax.legend(loc='upper center', bbox_to_anchor=(0.5,1.1), ncol=len(multiply_star_arr), prop={'size' : 12})

# solar rotation period
Prot_sun = 28   # in days on an average
solar_RDR = 7.0/441.0
plt.plot(Prot_sun, solar_RDR, 'k*', markersize=15)


# setting axis labels
ax.set_xlabel('$P_{\mathrm{rot}}$ in days')
ax.set_ylabel('$\\frac{\Delta\Omega}{\Omega}$', rotation=0, labelpad=15)

plt.tight_layout()

plt.savefig('KR98_inv_1_times_10stars.pdf')
    
    
