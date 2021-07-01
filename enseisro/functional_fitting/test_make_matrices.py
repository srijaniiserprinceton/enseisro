import numpy as np
from enseisro import globalvars
import enseisro.misc_functions as FN
from enseisro.synthetics import create_synth_modes as make_modes 
import matplotlib.pyplot as plt

ARGS = FN.create_argparser()
GVAR = globalvars.globalVars(ARGS)

# defining the multiplets
mults = np.array([[2,10], [2,12], [3,14]])

make_modes.make_modes(mults)
