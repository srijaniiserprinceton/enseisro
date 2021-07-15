import matplotlib.pyplot as plt
from enseisro import globalvars
from enseisro import misc_functions as FN

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
    
    


plot_solar_multiplets()
