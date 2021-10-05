import matplotlib.pyplot as plt
from enseisro import globalvars
from enseisro import misc_functions as FN
import numpy as np

GVAR = globalvars.globalVars()

font = {'size'   : 16}

plt.rc('font', **font)

def plot_solar_multiplets():
    """Plots the multiplet in the ell vs nu diagram.
    """
    mults = GVAR.nl_all

    plt.figure()
    
    for n in range(30):
        mask_mults = (GVAR.nl_all[:,0] == n)
        mults_n = mults[mask_mults]

        ells = mults_n[:,1]
            
        mult_idx = FN.nl_idx_vec(GVAR, mults_n)
        # omeganl in mHz
        omeganl = GVAR.omega_list[mult_idx] * (GVAR.OM * 1e3)

        plt.plot(ells, omeganl, 'ok', markersize=1)
        
    plt.xlabel('$\ell$')
    plt.ylabel('$\\nu$ in mHz')

    plt.savefig('nu_vs_ell_helio.pdf')
    
    plt.figure()
    
    for n in range(30):
        mask_mults = (GVAR.nl_all[:,0] == n)
        mults_n = mults[mask_mults]

        ells = mults_n[:,1]
            
        mult_idx = FN.nl_idx_vec(GVAR, mults_n)
        # omeganl in mHz
        omeganl = GVAR.omega_list[mult_idx] * (GVAR.OM * 1e3)

        plt.plot(ells[2:], omeganl[2:], 'ok', markersize=1, alpha=0.2)
        plt.plot(ells[:2], omeganl[:2], 'or', markersize=1)
        
        
    plt.xlabel('$\ell$')
    plt.ylabel('$\\nu$ in mHz')

    plt.savefig('nu_vs_ell_astero.pdf')
    
    
def plot_step_function():
    """Plots the step function for simple 
    demonstration
    """
    plt.figure(figsize=(10,6))

    r = np.linspace(0,1,100)
    rcz = 0.7
    rcz_ind = np.argmin(np.abs(r - rcz))
    
    step_function = np.zeros_like(r)
    step_function[r<rcz] = 1.0

    plt.plot(r[:rcz_ind], step_function[:rcz_ind], 'k', linewidth=4)
    plt.plot(r[rcz_ind+1:], step_function[rcz_ind+1:], 'k', linewidth=4)

    # customising the yticks and the locations
    yticks = np.array([0, 1])
    ylabels = np.array(['$\Omega^{\mathrm{out}}_s$','$\Omega^{\mathrm{in}}_s$'])

    plt.xlabel('$r$ in $R_{\odot}$')
    plt.ylabel('$\Omega_s(r)$')
    plt.yticks(ticks=yticks,labels=ylabels)

    # putting the double ended arrow to denote \Delta \Omega
    plt.annotate(text='', xy=(0.7,1), xytext=(0.7,0), arrowprops=dict(arrowstyle='<->', color='red'))
    plt.text(0.72, 0.5, '$\Delta \Omega_s$', color='red')
    plt.xlim([0,1])
    plt.ylim([-0.3, 1.3])

    plt.grid(True)

    plt.savefig('Step_function.pdf')


plot_solar_multiplets()
plot_step_function()
