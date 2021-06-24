import jax.numpy as jnp
import numpy as np
import enseisro.misc_functions as FN
from enseisro import globalvars

ARGS = FN.create_argparser()
GVAR = globalvars.globalVars(ARGS)

def get_eig(mode_idx):
    try:
        U = np.loadtxt(f'{GVAR.eigdir}/' +
                       f'U{mode_idx}.dat')[GVAR.rmin_idx:GVAR.rmax_idx]
        V = np.loadtxt(f'{GVAR.eigdir}/' +
                       f'V{mode_idx}.dat')[GVAR.rmin_idx:GVAR.rmax_idx]
    except FileNotFoundError:
        LOGGER.info('Mode file not found for mode index = {}'\
                    .format(mode_idx))
        return None
    return U, V
