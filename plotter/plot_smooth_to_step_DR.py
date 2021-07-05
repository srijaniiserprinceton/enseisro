import numpy as np
from enseisro import globalvars
import enseisro.misc_functions as FN
from enseisro import forward_functions_Omega as forfunc_Om
from enseisro.synthetics import w_omega_functions as w_om_func
from enseisro.synthetics import create_synthetic_DR as create_synth_DR
import matplotlib.pyplot as plt
import sys


ARGS = FN.create_argparser()
GVAR = globalvars.globalVars(ARGS)

# extracting the solar DR profile in terms of wsr                                                                                                                                  
wsr = create_synth_DR.get_solar_DR(GVAR, smax=3)

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

# calculating the step function equivalent of this Omegasr
step_param_arr = create_synth_DR.get_solar_stepfn_params(Omegasr, rcz_ind_arr)   # shape (Nstars x s x Nparams)
Omegasr_step = create_synth_DR.params_to_step(GVAR, step_param_arr, rcz_ind_arr) # shape (Nstars x s x r)

# converting to nHz
Omegasr = Omegasr * GVAR.OM * 1e9
Omegasr_step = Omegasr_step * GVAR.OM * 1e9

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
        ax[s_ind].plot(GVAR.r, Omegasr_step[star_ind,s_ind,:], '--r', label='Star=%i, s=%i, Step'%(star_ind+1,2*s_ind+1)) 
        ax[s_ind].set_ylabel('$\Omega_{%i}(r)$ in nHz'%(2*s_ind+1))
    ax[s_ind].legend()
    ax[s_ind].grid(True, alpha=0.5)

ax[1].set_xlabel('$r$ in $R_{\odot}$')


# getting the frequency splitting induced by the smooth and step-function
mults = np.array([[2,10]]) # ,[3,2],[3,6],[2,8]])
domega_smooth = forfunc_Om.compute_splitting(GVAR, Omegasr[0], mults)
domega_step = forfunc_Om.compute_splitting(GVAR, Omegasr_step[0], mults)

# creating the cumulative mode arr for plotting.  Summing over (2*ell + 1)
cumulative_modes = np.arange(0, 2*np.sum(mults[:,1])+len(mults))
# labels for plotting
modeind_start, modeind_end = 0,0
for i, mult in enumerate(mults):
    # getting the frequency splitting induced by the smooth and step-function                                                                                                           
    domega_smooth = forfunc_Om.compute_splitting(GVAR, Omegasr[0], np.array([mult]))
    domega_step = forfunc_Om.compute_splitting(GVAR, Omegasr_step[0], np.array([mult]))
    
    # converting to nHz. Does not need GVAR.OM*1e9 since the Omegasr and Omegasr_step are 
    # already converted to nHz before computing the splittings
    domega_smooth *= 1
    domega_step *= 1

    # getting the central freq omeganl for the mode
    mult_idx = FN.nl_idx(GVAR, mult[0], mult[1])                                                                                                                                        
    omeganl = GVAR.omega_list[mult_idx]                    # not in nHz                                                                                                                 

    # adding to get true values of omeganlm
    omeganlm_smooth = domega_smooth + (omeganl * GVAR.OM * 1e9)
    omeganlm_step = domega_step + (omeganl * GVAR.OM * 1e9)

    # creating 2*ell+1 spaces for plotting
    modeind_end = modeind_start + (2 * mult[1] + 1)
    
    # All value are in nHz till now. Converting and plotting in  mHz
    omeganlm_smooth *= 1e-6
    omeganlm_step *= 1e-6

    if(i == 0):
        ax_big.plot(cumulative_modes[modeind_start:modeind_end], omeganlm_smooth, '.k', label='Smooth', alpha=1.0)
        ax_big.plot(cumulative_modes[modeind_start:modeind_end], omeganlm_step, 'xr', label='Step', alpha=0.7)
        #ax_big.plot(cumulative_modes[modeind_start:modeind_end], omeganlm_smooth-omeganlm_step, '.k', label='Sm\
#ooth', alpha=1.0)
    else:
        ax_big.plot(cumulative_modes[modeind_start:modeind_end], omeganlm_smooth, '.k', alpha=1.0)
        ax_big.plot(cumulative_modes[modeind_start:modeind_end], omeganlm_step, 'xr', alpha=0.7)

    modeind_start = modeind_end

ax_big.set_ylabel('$\omega_{n \ell m}$ in mHz')
ax_big.set_xlabel('Cumulative mode label')
ax_big.grid(True, alpha=0.5)
ax_big.legend()

plt.tight_layout()
plt.savefig('DR_smooth_and_step.pdf')
