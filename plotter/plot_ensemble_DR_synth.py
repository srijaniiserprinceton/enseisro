# import jax.numpy as np
import numpy as np
from enseisro import globalvars
import enseisro.misc_functions as FN
from enseisro.synthetics import create_synthetic_DR as create_syn_DR
from enseisro.synthetics import create_rot_prof_KR98 as create_KR98_prof
from enseisro.synthetics import w_omega_functions as w_om_func
import matplotlib.pyplot as plt
from enseisro.synthetics import create_synthetic_DR as create_synth_DR
from scipy.integrate import simpson as simp
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ARGS = FN.create_argparser()
# GVAR = globalvars.globalVars(ARGS)

GVAR = globalvars.globalVars()

NAX = np.newaxis

font = {'size'   : 16}
plt.rc('font', **font)

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

    # converting to Omegasr of shape (s x r x Nstars)                                                                      
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

    # plotting radial after averaging over theta (bulk rotation)
    Omega_r_bulk = simp(2 * 0.75 * Omega_r_theta_true[rmin_ind:,:] * (np.sin(theta)**3)[np.newaxis,:], x=theta, axis=1)
    
    plt.figure()
    plt.plot(GVAR.r[rmin_ind:], Omega_r_bulk, 'k')
    plt.xlabel('$r$ in $R_{\odot}$')
    plt.ylabel('$\Omega(r)$')
    
    plt.savefig('Omega_bulk.pdf')


def plot_Omega_2D_step(N=10, p=1, rmin_plot=0.3, Ntheta=100):
    Nstars = N

    # getting the ensemble w_s(r)                                                                                                                                                               
    wsr, wsr_ens = get_ens_synth_DR(N, p)

    # converting to Omegasr of shape (s x r x Nstars)                                                                                                                                            
    Omegasr = w_om_func.w_2_omega(GVAR, wsr)
    Omegasr_ens = w_om_func.w_2_omega(GVAR, wsr_ens)

    lens, lenr = Omegasr.shape

    # converting to shape (Nstars x s x r)                                                                                                                                                       
    Omegasr_ens = np.swapaxes(Omegasr_ens, 0, 2)
    Omegasr_ens = np.swapaxes(Omegasr_ens, 1, 2)


    # Base of the convection zone for each star                                                                                                                                                  
    rcz_arr = np.zeros(Nstars) + 0.7

    # finding the rcz index for all the stars                                                                                                                                                    
    rcz_ind_arr = np.zeros(Nstars, dtype='int')
    for i in range(Nstars):
        rcz_ind = np.argmin(np.abs(GVAR.r - rcz_arr[i]))
        rcz_ind_arr[i] = rcz_ind

    step_param_arr_in_out = create_synth_DR.get_solar_stepfn_params(np.reshape(Omegasr,(1,lens,lenr)), rcz_ind_arr)   # shape (Nstars x s x  Nparams)
    # calculating the step function equivalent of this Omegasr. (\Omega_{in} + \Omega_{out))                                                                                                     
    step_param_arr_in_out_ens = create_synth_DR.get_solar_stepfn_params(Omegasr_ens, rcz_ind_arr)   # shape (Nstars x s x  Nparams)                                                                  

    # converting back to step function in r for plotting. (Nstars x s x r)                                                                                                                      
    Omegasr = create_synth_DR.params_to_step(GVAR, step_param_arr_in_out, rcz_ind_arr)[0]
    Omegasr_ens = create_synth_DR.params_to_step(GVAR, step_param_arr_in_out_ens, rcz_ind_arr)

    # converting back to shape (s x r x Nstars)                                                                                                                                                 
    Omegasr_ens = np.swapaxes(Omegasr_ens, 0, 2)
    Omegasr_ens = np.swapaxes(Omegasr_ens, 0, 1)


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

    plt.savefig('Omega_2D_ens_step.pdf')

def plot_Omega_2D_KSPAReport(Prot_arr, p=0, rmin_plot=0.3, Ntheta=100):
    Nstars = N = len(Prot_arr)
    Nparams = 2

    # getting the solar w_s(r)                                                                                                                                                           
    wsr, __ = get_ens_synth_DR(1, p)

    # converting to Omegasr of shape (s x r x Nstars)                                                                                                                                            
    Omegasr = w_om_func.w_2_omega(GVAR, wsr)
    lens, lenr = Omegasr.shape

    step_param_arr_in_out_arr = np.zeros((Nstars, lens, Nparams))
    
    # getting the step parameters for the Nstars. Shape of arrays (Nstars, lens) in nHz
    Om_in_arr, Om_out_arr = create_KR98_prof.get_step_params_from_Prot(Prot_arr) 

    # converting to non-dimensional parameters. Adding negative to have same convention
    Om_out_arr = - Om_out_arr / (GVAR.OM * 1e9)
    Om_in_arr = - Om_in_arr / (GVAR.OM * 1e9)

    # filling in the step params
    step_param_arr_in_out_arr[:,:,0] = Om_in_arr
    step_param_arr_in_out_arr[:,:,1] = Om_out_arr

    print(step_param_arr_in_out_arr * GVAR.OM * 1e9)

    # Base of the convection zone for each star                                                                                                                                                  
    rcz_arr = np.zeros(Nstars) + 0.7

    # finding the rcz index for all the stars                                                                                                                                                    
    rcz_ind_arr = np.zeros(Nstars, dtype='int')
    for i in range(Nstars):
        rcz_ind = np.argmin(np.abs(GVAR.r - rcz_arr[i]))
        rcz_ind_arr[i] = rcz_ind


    # getting the step parameters of the Sun
    step_param_arr_in_out = create_synth_DR.get_solar_stepfn_params(np.reshape(Omegasr,(1,lens,lenr)), rcz_ind_arr)
    # setting the Omega_in_3 to zero
    step_param_arr_in_out[0,1,0] = 0.0

    print(step_param_arr_in_out * GVAR.OM * 1e9)
    # converting back to step function in r for plotting. (Nstars x s x r)               
    Omegasr = create_synth_DR.params_to_step(GVAR, step_param_arr_in_out, rcz_ind_arr)[0]
    Omegasr_arr = create_synth_DR.params_to_step(GVAR, step_param_arr_in_out_arr, rcz_ind_arr)

    # converting back to shape (s x r x Nstars)                                                                                                                                                 
    Omegasr_arr = np.swapaxes(Omegasr_arr, 0, 2)
    Omegasr_arr = np.swapaxes(Omegasr_arr, 0, 1)


    # index for rmin_plot                                                                                                                                                                        
    rmin_ind = np.argmin(np.abs(GVAR.r - rmin_plot))
    r = GVAR.r[rmin_ind:]

    # getting the Omega(r,theta) profile                                                                                                                                                         
    Omega_r_theta, theta = w_om_func.Omega_s_to_Omega_theta(Omegasr_arr,\
                                                            Ntheta=Ntheta)
    Omega_r_theta_true, theta = w_om_func.Omega_s_to_Omega_theta(Omegasr,Ntheta=Ntheta)
   

    # getting them to correct units (nHz)                                                                                                                                                        
    Omega_r_theta_true *= GVAR.OM * 1e9
    Omega_r_theta *= GVAR.OM * 1e9

    # creating the meshgrid                                                                                                                                                                      
    rr,tt = np.meshgrid(r, theta, indexing='ij')
    # making a slight adjsustment since we are using co-latitude                                                                                                                                 
    # in defining theta and not latitude                                                                                                                                                         
    xx = rr * np.cos(np.pi/2. - tt)
    yy = rr * np.sin(np.pi/2. - tt)

    # creating the grid of subplots                                                                                                                                                              
    nrows, ncols = 2, 2
    
    #fig, axs = plt.subplots(nrows, ncols, figsize=(15,6))                                                                                                                                       
    fig, ax = plt.subplots(nrows, ncols, figsize=(10,8))

    # colormap vmin and vmax                                                                                                                                                                      
    vmin, vmax = np.amin(Omega_r_theta_true), np.amax(Omega_r_theta_true)
    vmin *= 0.5
    vmax *= 5.0

    # choosing the max of abs of vmax and vmin since we want                                                                                                                                      
    # white to be about 0.0                                                                                                                                                                       
    vmaxabs = max(np.abs(vmin),np.abs(vmax))

    # drawing the inner and outer surface lines (just for cleanliness)                                                                                                                            
    r_out, r_in = 1.0, rmin_plot
    x_out, y_out = r_out * np.cos(theta), r_out * np.sin(theta)
    x_in, y_in = r_in * np.cos(theta), r_in * np.sin(theta)


    # plotting the solar profile
    im = ax[0,0].pcolormesh(xx, yy, -Omega_r_theta_true[rmin_ind:,:],\
                            rasterized=True, cmap='autumn')
    ax[0,0].plot(x_in, y_in, 'k')
    ax[0,0].plot(x_out, y_out, 'k')
    ax[0,0].set_aspect('equal')
    ax[0,0].text(0.6, 0.9, '$\Omega_{\odot}$ in nHz')
    
    divider = make_axes_locatable(ax[0,0])
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(im, cax=cax)

    # Omega ratio with respect to solar Omega
    Omega_r_theta_norm = Omega_r_theta / Omega_r_theta_true[:,:,np.newaxis]

    # star label
    star_label = np.array(['A', 'B', 'C'])

    # looping over the cases in the ensemble                                    
    for i in range(0, nrows):
        for j in range(0, ncols):
            if(i==0 and j==0): continue
            print(Omega_r_theta[:,:,2*i+j-1]/Omega_r_theta_true)
            im = ax[i,j].pcolormesh(xx, yy, Omega_r_theta_norm[rmin_ind:,:,(2*i+j)-1],\
                                    rasterized=True, vmin = 0, vmax = 8, cmap='terrain')
            ax[i,j].plot(x_in, y_in, 'k')
            ax[i,j].plot(x_out, y_out, 'k')
            ax[i,j].set_aspect('equal')

            ax[i,j].text(0.6, 0.9, '$\Omega_{\mathrm{%s}} / \Omega_{\odot}$'%star_label[2*i+j-1])

            divider = make_axes_locatable(ax[i,j])
            cax = divider.append_axes("right", size="5%", pad=0.05)

            fig.colorbar(im, cax = cax)

    plt.tight_layout()

    plt.savefig('Omega_2D_KSPA_Report.pdf')


# plot_ens_DR()

#plot_Omega_2D(p=1)
#plot_Omega_2D_step(p=1)

# Prot array in days
Prot_arr = np.array([5, 10, 25])
plot_Omega_2D_KSPAReport(Prot_arr)
