# import jax.numpy as np
import numpy as np
from enseisro import globalvars
import enseisro.misc_functions as FN
from enseisro.synthetics import create_synthetic_DR as create_syn_DR
from enseisro.synthetics import w_omega_functions as w_om_func
import matplotlib.pyplot as plt

ARGS = FN.create_argparser()
GVAR = globalvars.globalVars(ARGS)

NAX = np.newaxis

def get_ens_synth_DR(N=10, p=1):
    # getting the solar differential rotation profile
    wsr = create_syn_DR.get_solar_DR(GVAR)
    # wsr[1,:] *= 0.0

    # making N copies with p-percent change at most
    wsr_ens = create_syn_DR.make_DR_copies_random(GVAR, wsr, N=N, p=p)

    return wsr, wsr_ens

def plot_ens_DR(N=10, p=1, rmin_plot=0.3):
    fig, ax = plt.subplots(1, 2, figsize=(12,6))
    
    # getting the ensemble w_s(r)
    wsr, wsr_ens = get_ens_synth_DR(N, p)

    # converting to Omegasr
    wsr = w_om_func.w_2_omega(GVAR, wsr)
    wsr_ens = w_om_func.w_2_omega(GVAR, wsr_ens)

    # index for rmin_plot
    rmin_ind = np.argmin(np.abs(GVAR.r - rmin_plot))

    for i in range(N):
        # w_1(r)
        ax[0].plot(GVAR.r[rmin_ind:], wsr_ens[0,rmin_ind:,i], '--')
        # w_3(r)
        ax[1].plot(GVAR.r[rmin_ind:], wsr_ens[1,rmin_ind:,i], '--')

    # plotting the true profile first                                                             
    ax[0].plot(GVAR.r[rmin_ind:], wsr[0,rmin_ind:], 'k')
    ax[1].plot(GVAR.r[rmin_ind:], wsr[1,rmin_ind:], 'k')
    

    ax[0].set_ylabel('$w_{%i}(r)$'%1)
    ax[1].set_ylabel('$w_{%i}(r)$'%3)
    ax[0].set_xlabel('$r$ in $R_*$')
    ax[1].set_xlabel('$r$ in $R_*$')
    
    plt.tight_layout()

    plt.savefig('DR_synth_ens.pdf')


def plot_Omega_2D(N=10, p=1, rmin_plot=0.3, Ntheta=100):
    # getting the ensemble w_s(r)                                                                
    wsr, wsr_ens = get_ens_synth_DR(N, p)

    # converting to Omegasr                                                                      
    Omegasr = w_om_func.w_2_omega(GVAR, wsr)
    Omegasr_ens = w_om_func.w_2_omega(GVAR, wsr_ens)

    # index for rmin_plot                                                                      
    rmin_ind = np.argmin(np.abs(GVAR.r - rmin_plot))
    r = GVAR.r[rmin_ind:]

    # getting the Omega(r,theta) profile
    Omega_r_theta, theta = w_om_func.Omega_s_to_Omega_theta(Omegasr_ens,\
                                                            Ntheta=Ntheta)
    Omega_r_theta_true, theta = w_om_func.Omega_s_to_Omega_theta(Omegasr,Ntheta=Ntheta)

    # creating the meshgrid
    rr,tt = np.meshgrid(r, theta, indexing='ij')
    xx = rr * np.cos(tt)
    yy = rr * np.sin(tt)
    
    # creating the grid of subplots
    nrows, ncols = 2, 5

    fig, axs = plt.subplots(nrows, ncols, figsize=(15,6))

    # colormap vmin and vmax
    # vmin, vmax = np.amin(Omega_r_theta_true), np.amax(Omega_r_theta_true)
    # vmin *= 1.1
    # vmax *= 1.1

    # looping over the cases in the ensemble
    for i in range(N):
        row, col = i//5, i%5
        axs[row,col].pcolormesh(xx, yy, Omega_r_theta[rmin_ind:,:,i],\
                                rasterized=True) #, vmin=vmin, vmax=vmax)

    plt.tight_layout()

    plt.savefig('Omega_2D_ens.pdf')

# plot_ens_DR()
plot_Omega_2D(p=0)
