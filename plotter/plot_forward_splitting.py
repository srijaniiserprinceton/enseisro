import numpy as np
from enseisro import globalvars
import enseisro.misc_functions as FN
import enseisro.forward_functions as forfunc
import matplotlib.pyplot as plt

ARGS = FN.create_argparser()
GVAR = globalvars.globalVars(ARGS)
n, ell = 2, 10
mult = np.array([[ell,n]])
sarr = np.array([1,3,5])

domega_m = forfunc.compute_splitting(GVAR, mult, sarr)
m_arr = np.arange(-mult[0][0], mult[0][0]+1)

plt.plot(m_arr, domega_m ,'ok')
plt.xlabel('$m$')
plt.ylabel('$\delta \omega^{%i,%i}(m)$'%(n,ell))
plt.tight_layout()
plt.savefig('freq_splitting.pdf')
