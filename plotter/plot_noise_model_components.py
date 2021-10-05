import numpy as np
import matplotlib.pyplot as plt
from enseisro.noise_model import misc_noise_functions as Noise_FN

def plot_N_nu():
    """This function plots the background noise alone composed of the two
    Harvey models. Should be identical to Fig.2.4 in Stahn's thesis."""
    
    # values of the parameters taken from Table 2.1 in Stahn's thesis.
    A_arr = np.array([1.607, 0.542])       # amplitude array in ppm^2 \muHz^{-1}
    A_err_arr = np.array([0.082, 0.030])   # error in amplitude array in ppm^2 \muHz^{-1}

    tau_arr = np.array([1390.0, 455.0])    # time-scale array in seconds
    tau_arr_err = np.array([30, 10])       # error in time-scale array in seconds

    # photon white noise in ppm^2 \muHz^{-1}
    P_wn = 0.00065    # this is approx value Stahn's Fig 2.4 has. But he says that P_wn < 0.004.
    
    # definiing the frequency array in muHz
    nu_arr = np.logspace(2,4)


    N_nu_arr, harvey_models = Noise_FN.make_N_nu(nu_arr, tau_arr, A_arr, P_wn, return_harveys=True)


    # plotting the background noise N(\nu)
    plt.figure()
    
    # plotting background = harvey_1 + harvey_2
    plt.loglog(nu_arr, N_nu_arr, '--b', label='$\\mathcal{N(\\nu)}$')

    # plotting the individual harvey models
    plt.loglog(nu_arr, harvey_models[0], '--y', label='Harvey 1')
    plt.loglog(nu_arr, harvey_models[1], '--y', label='Harvey 2')

    # plotting the photon noise
    plt.loglog(nu_arr, np.zeros_like(nu_arr) + P_wn, '--k', label='Photon white noise')

    plt.legend()

    plt.xlabel('Frequency [$\mu$Hz]')
    plt.ylabel('Power [$\mathrm{ppm}^2/\mu$Hz]')

    plt.xlim([1e2, 1e4])
    plt.ylim([1e-4, 1e2])
    
    plt.tight_layout()

    plt.savefig('background_noise.pdf')
    plt.close()

# plotting the sigma vs freq
def plot_sigma_vs_freq():
    """Plots the uncertainty obtained from the                                       
    the noise model vs the mode frequencies.  

    Parameters                                                                                
    ----------                                                                              
    sigma_arr : array_like, float                                                           
        Array containing the noise in nHz for all modes.                                    
    omega_nlm_arr : array_like, float                                                       
        Array containing omega_nlm in muHz for all modes.                                   
    """
    # reading from the stored files for ell = 1, 2
    mode_freq_arr_l1 = np.load('mode_freq_arr_l1.npy')
    mode_freq_arr_l2 = np.load('mode_freq_arr_l2.npy')

    sigma_arr_l1 = np.load('sigma_arr_l1.npy')
    sigma_arr_l2 = np.load('sigma_arr_l2.npy')


    sort_args_mode_freq_l1 = np.argsort(mode_freq_arr_l1)
    sigma_arr_l1 = sigma_arr_l1[sort_args_mode_freq_l1]
    mode_freq_arr_l1 = mode_freq_arr_l1[sort_args_mode_freq_l1]

    sort_args_mode_freq_l2 = np.argsort(mode_freq_arr_l2)
    sigma_arr_l2 = sigma_arr_l2[sort_args_mode_freq_l2]
    mode_freq_arr_l2 = mode_freq_arr_l2[sort_args_mode_freq_l2]

    plt.figure(figsize=(10,6))

    plt.plot(mode_freq_arr_l1, sigma_arr_l1, 'k+', markersize=10, label='$\ell=1$')
    plt.plot(mode_freq_arr_l1, sigma_arr_l1, 'k')

    plt.plot(mode_freq_arr_l2, sigma_arr_l2, 'r+', markersize=10, label='$\ell=2$')
    plt.plot(mode_freq_arr_l2, sigma_arr_l2, 'r')

    plt.xlabel('$\omega_{nlm}$ in $\mu$Hz')
    plt.ylabel('$\sigma(\delta \omega_{nlm}$) in nHz')

    plt.legend()
    plt.grid(True)
    
    plt.savefig('sigma_vs_freq.pdf')
    plt.close()

# plotting and saving the background noise
plot_N_nu()

plot_sigma_vs_freq()
