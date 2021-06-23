import jax.numpy as jnp
import numpy as np

ARGS = FN.create_argparser()
GVAR = globalvars.globalVars(ARGS)

def get_eig(self, mode_idx):
    try:
        U = np.loadtxt(f'{self.sup.gvar.eigdir}/' +
                       f'U{mode_idx}.dat')[self.rmin_idx:self.rmax_idx]
        V = np.loadtxt(f'{self.sup.gvar.eigdir}/' +
                       f'V{mode_idx}.dat')[self.rmin_idx:self.rmax_idx]
    except FileNotFoundError:
        LOGGER.info('Mode file not found for mode index = {}'\
                    .format(mode_idx))
        return None
    return U, V
