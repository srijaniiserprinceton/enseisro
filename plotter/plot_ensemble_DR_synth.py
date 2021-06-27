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

    # we shall be plotting the difference from the true profile
    Omega_diff = Omega_r_theta - Omega_r_theta_true[:,:,NAX]

    # getting them to correct units (nHz)
    Omega_r_theta_true *= GVAR.OM * 1e9
    Omega_diff *= GVAR.OM * 1e9

    # creating the meshgrid
    rr,tt = np.meshgrid(r, theta, indexing='ij')
    # making a slight adjsustment since we are using co-latitude
    # in defining theta and not latitude
    xx = rr * np.cos(np.pi/2. - tt)
    yy = rr * np.sin(np.pi/2. - tt)
    
    # creating the grid of subplots
    nrows, ncols = 2, 5

    #fig, axs = plt.subplots(nrows, ncols, figsize=(15,6))
    fig = plt.figure(figsize=(21,6))
    
    gs = fig.add_gridspec(2,7)
    ax = []
    for i in range(N):
        row, col = i//5, i%5
        ax.append(fig.add_subplot(gs[row,2+col]))

    ax_big = fig.add_subplot(gs[0:2,0:2])

    # colormap vmin and vmax
    vmin, vmax = np.amin(Omega_diff), np.amax(Omega_diff)
    vmin *= 1.1
    vmax *= 1.1

    # choosing the max of abs of vmax and vmin since we want 
    # white to be about 0.0
    vmaxabs = max(np.abs(vmin),np.abs(vmax))

    # drawing the inner and outer surface lines (just for cleanliness)                           
    r_out, r_in = 1.0, rmin_plot
    x_out, y_out = r_out * np.cos(theta), r_out * np.sin(theta)
    x_in, y_in = r_in * np.cos(theta), r_in * np.sin(theta)

    # looping over the cases in the ensemble
    for i in range(N):
        im = ax[i].pcolormesh(xx, yy, Omega_diff[rmin_ind:,:,i],\
        rasterized=True, vmin=-vmaxabs, vmax=vmaxabs, cmap='seismic')
        ax[i].plot(x_in, y_in, 'k')
        ax[i].plot(x_out, y_out, 'k')
        ax[i].set_aspect('equal')

    # fig.subplots_adjust(left=0.0, right=0.95)
    plt.subplots_adjust(left=0.01, right=0.95, bottom=0.15, top=0.98, wspace=0.1, hspace=0.2)
    # put colorbar at desire position
    cbar_ax = fig.add_axes([0.97, 0.14, 0.005, 0.84])
    fig.colorbar(im, cax=cbar_ax)

    ax[7].set_xlabel('$\Omega_{*}(\\theta,\phi) - \Omega_{\odot}(\\theta,\phi)$ in nHz', \
                     labelpad = 15, fontsize=16)
    
    im = ax_big.pcolormesh(xx, yy, Omega_r_theta_true[rmin_ind:,:], rasterized=True)
    ax_big.plot(x_in, y_in, 'k')
    ax_big.plot(x_out, y_out, 'k')
    ax_big.set_aspect('equal')
    ax_big.text(0.65,0.85,'$\Omega_{\odot}(\\theta,\phi)$ in nHz',fontsize=16)

    cbar_ax = fig.add_axes([0.014, 0.05, 0.25, 0.01])
    fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    # fig.colorbar(im, ax=ax_big, orientation='horizontal')

    # plt.tight_layout()

    plt.savefig('Omega_2D_ens.pdf')

# plot_ens_DR()
plot_Omega_2D(p=1)
