import numpy as np
from enseisro import globalvars
import enseisro.misc_functions as FN
from enseisro import forward_functions_Omega as forfunc_Om
from enseisro.synthetics import w_omega_functions as w_om_func
from enseisro.synthetics import create_synthetic_DR as create_synth_DR
from enseisro.functional_fitting import a_solver as a_solver
import matplotlib.pyplot as plt
from enseisro.synthetics import create_synth_modes as make_modes
import sys

font = {#'family' : 'normal',                                                                                                                                                                    
        #'weight' : 'bold',                                                                                                                                                                      
        'size'   : 16}

plt.rc('font', **font)

# ARGS = FN.create_argparser()
# GVAR = globalvars.globalVars(ARGS)

GVAR = globalvars.globalVars()

smax = 3

# extracting the solar DR profile in terms of wsr                                                                                                                                  
wsr = create_synth_DR.get_solar_DR(GVAR, smax=smax)

# converting this wsr to Omegasr                                                                                                                                                       
Omegasr = w_om_func.w_2_omega(GVAR, wsr)
lens = Omegasr.shape[0]

# number of stars
Nstars = 1

# finding the rcz index
rcz = 0.7
rcz_ind = np.argmin(np.abs(GVAR.r - rcz))
rcz_ind_arr = np.array([rcz_ind])   # contains only one star

# reshaping Omegasr to give it (Nstars x s x r) shape
Omegasr = np.reshape(Omegasr, (Nstars, lens, len(GVAR.r)))

print(Omegasr * GVAR.OM * 1e9)

# calculating the step function equivalent of this Omegasr
step_param_arr = create_synth_DR.get_solar_stepfn_params(Omegasr, rcz_ind_arr)   # shape (Nstars x s x Nparams)
# the Omegasr step from usual averaging
Omegasr_step_avg = create_synth_DR.params_to_step(GVAR, step_param_arr, rcz_ind_arr) # shape (Nstars x s x r)

print(Omegasr_step_avg * GVAR.OM * 1e9)

# creating modes for inversion
nmin, nmax = 16, 23
lmin, lmax = 2, 2


mults_single_star_lo_ell = FN.build_mults_single_star(nmin, nmax, lmin, lmax)
'''
nmin, nmax = 0, 2
lmin, lmax = 150, 200
mults_single_star_hi_ell = FN.build_mults_single_star(nmin, nmax, lmin, lmax)

mults_single_star = np.append(mults_single_star_lo_ell, mults_single_star_hi_ell, axis=0)
'''
mults_single_star = mults_single_star_lo_ell


# getting the modes for which we want splitting values for one star                                                                                                                  
# these would later be repeated to make multiple stars                                                                                                                               
modes_single_star = make_modes.make_modes(mults_single_star)

# inversion using the smooth profile to get step params arrs
sigma_arr = np.ones(modes_single_star.shape[1])

# step param array from inversion
step_param_arr_inv, __ = a_solver.use_numpy_inv_Omega_function(GVAR, modes_single_star, sigma_arr, smax, Omegasr=Omegasr_step_avg[0])

print(step_param_arr_inv * GVAR.OM * 1e9)

# converting back to (Nstars x s x r) shape
step_param_arr_inv = np.reshape(step_param_arr_inv, (Nstars, lens, 2))
# Omegasr step from inversion where we use smooth Omegasr to generate data and invert for step
Omegasr_step_inv = create_synth_DR.params_to_step(GVAR, step_param_arr_inv, rcz_ind_arr)

print(step_param_arr_inv * GVAR.OM * 1e9)

# converting to nHz
Omegasr = Omegasr * GVAR.OM * 1e9
Omegasr_step_avg = Omegasr_step_avg * GVAR.OM * 1e9
Omegasr_step_inv = Omegasr_step_inv * GVAR.OM * 1e9

print(Omegasr_step_inv)

# creating subplots
# fig, ax = plt.subplots(2, 2, figsize=(10,6), sharex = True)
fig = plt.figure(figsize=(16,6))
gs = fig.add_gridspec(2,2)
ax = []
ax.append(fig.add_subplot(gs[0,0]))
ax.append(fig.add_subplot(gs[1,0]))
    
ax_big = fig.add_subplot(gs[:,1])

# plotting to compare the smooth DR and step function DR
for s_ind in range(lens):
    for star_ind in range(Nstars):
        ax[s_ind].plot(GVAR.r, Omegasr[star_ind,s_ind,:], 'k', label='Star=%i, s=%i, Smooth'%(star_ind+1,2*s_ind+1))
        ax[s_ind].plot(GVAR.r[::50], Omegasr_step_avg[star_ind,s_ind,::50], '.-b', label='Star=%i, s=%i, Averaged step function'%(star_ind+1,2*s_ind+1))
        ax[s_ind].plot(GVAR.r, Omegasr_step_inv[star_ind,s_ind,:], '--r', label='Star=%i, s=%i, Inverted step function'%(star_ind+1,2*s_ind+1)) 
        ax[s_ind].set_ylabel('$\Omega_{%i}(r)$ in nHz'%(2*s_ind+1))
    ax[s_ind].legend(prop={'size' : 12})
    ax[s_ind].grid(True, alpha=0.5)

ax[1].set_xlabel('$r$ in $R_{\odot}$')


# getting the frequency splitting induced by the smooth and step-function
mults = np.array([[20,3]]) # ,[3,2],[3,6],[2,8]])
# domega_smooth = forfunc_Om.compute_splitting_from_function(GVAR, Omegasr[0], mults)
# domega_step = forfunc_Om.compute_splitting_from_function(GVAR, Omegasr_step_inv[0], mults)

# creating the cumulative mode arr for plotting.  Summing over (2*ell + 1)
cumulative_modes = np.arange(0, 2*np.sum(mults[:,1])+len(mults))
# labels for plotting
modeind_start, modeind_end = 0,0
for i, mult in enumerate(mults):
    # getting the frequency splitting induced by the smooth and step-function                                                                                                           
    domega_smooth = forfunc_Om.compute_splitting_from_function(GVAR, Omegasr[0], np.array([mult]))
    domega_step = forfunc_Om.compute_splitting_from_function(GVAR, Omegasr_step_inv[0], np.array([mult]))
    
    # converting to nHz. Does not need GVAR.OM*1e9 since the Omegasr and Omegasr_step are 
    # already converted to nHz before computing the splittings
    domega_smooth *= 1
    domega_step *= 1

    # getting the central freq omeganl for the mode
    # mult_idx = FN.nl_idx(GVAR, mult[0], mult[1])                                                                                                                                        
    # omeganl = GVAR.omega_list[mult_idx]                    # not in nHz                                                                                                                 

    # adding to get true values of omeganlm
    # omeganlm_smooth = domega_smooth + (omeganl * GVAR.OM * 1e9)
    # omeganlm_step = domega_step + (omeganl * GVAR.OM * 1e9)

    # creating 2*ell+1 spaces for plotting
    modeind_end = modeind_start + (2 * mult[1] + 1)
    
    # All value are in nHz till now. Converting and plotting in  mHz
    # omeganlm_smooth *= 1e-6
    # omeganlm_step *= 1e-6

    inst_m_arr = np.arange(-mult[1], mult[1]+1)
    #print(inst_m_arr)
    #print(cumulative_modes

    '''
    if(i == 0):
        # ax_big.plot(inst_m_arr, cumulative_modes[modeind_start:modeind_end], domega_smooth, '.k', label='Smooth', alpha=1.0, markersize=15)
        # ax_big.plot(inst_m_arr, cumulative_modes[modeind_start:modeind_end], domega_step, 'xr', label='Step', alpha=0.7, markersize=15)
        ax_big.plot(inst_m_arr, domega_smooth, '.k', label='Smooth', alpha=1.0, markersize=15)
        ax_big.plot(inst_m_arr, domega_step, 'xr', label='Step', alpha=0.7, markersize=15)
        # ax_big.plot(inst_m_arr, domega_smooth, '.k', label='$\delta \omega_{n\ell m}^{\mathrm{smooth}} - \delta \omega_{n\ell m}^{\mathrm{step}}', alpha=1.0, markersize=15)
        
    else:
        # ax_big.plot(inst_m_arr, cumulative_modes[modeind_start:modeind_end], domega_smooth, '.k', alpha=1.0, markersize=15)
        # ax_big.plot(inst_m_arr, cumulative_modes[modeind_start:modeind_end], domega_step, 'xr', alpha=0.7, markersize=15)
        ax_big.plot(inst_m_arr, domega_smooth, '.k', alpha=1.0, markersize=15)
        ax_big.plot(inst_m_arr, domega_step, 'xr', alpha=0.7, markersize=15)
    '''
    ax_big.plot(inst_m_arr, domega_smooth-domega_step, '.k', label='$\delta \omega_{n\ell m}^{\mathrm{smooth}} - \delta \omega_{n\ell m}^{\mathrm{step}}$', alpha=1.0, markersize=15)

    modeind_start = modeind_end

ax_big.set_ylabel('$\delta \omega_{%i, %i, m}$ in nHz'%(mults[0,0],mults[0,1]))
ax_big.set_xlabel('$m$')
ax_big.grid(True, alpha=0.5)
ax_big.legend(prop={'size' : 12})

plt.tight_layout()
plt.savefig('DR_smooth_and_step.pdf')
