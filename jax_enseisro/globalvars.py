import numpy as np
import os
# import enseisro.misc_functions as FN

#----------------------------------------------------------------------
#                       All qts in CGS
# M_sol = 1.989e33 g
# R_sol = 6.956e10 cm
# B_0 = 10e5 G
# OM = np.sqrt(4*np.pi*R_sol*B_0**2/M_sol)
# rho_0 = M_sol/(4pi R_sol^3/3) = 1.41 ~ 1g/cc (for kernel calculation)
#----------------------------------------------------------------------
filenamepath = os.path.realpath(__file__)
# taking [:-2] since we are ignoring the file name and current dirname
# this is specific to the way the directory structure is constructed
filepath = '/'.join(filenamepath.split('/')[:-2])   
jax_enseisro_path = '/'.join(filenamepath.split('/')[:-1])
configpath = filepath
with open(f"{configpath}/.config", "r") as f:
    dirnames = f.read().splitlines()

# reading the start data paths stored in .star_data_paths
with open(f"{jax_enseisro_path}/.star_data_paths", "r") as f:
    star_data_paths = f.read().splitlines()

class qdParams():
    # {{{ Reading global variables
    # setting rmax as 1.2 because the entire r array needs to be used
    # in order to reproduce
    # (1) the correct normalization
    # (2) a1 = \omega_0 ( 1 - 1/ell ) scaling
    # (Since we are using lmax = 300, 0.45*300 \approx 150)
    rmin = 0.0
    rmax = 1.2
    fwindow =  150 
    years_obs = 3  # in years

class GlobalVars():
    def __init__(self, nStype=10, nmin=16, nmax=24, lmin=1, lmax=2,
                 smax=3, rand_mults=0, add_Noise=0, use_Delta=1,
                 metadata_path = '.'):
        # incorporating the params from metadata
        self.nStype = nStype
        self.nmin, self.nmax = nmin, nmax
        self.lmin, self.lmax = lmin, lmax
        self.smax = smax
        self.rand_mults = rand_mults
        self.add_Noise = add_Noise
        self.use_Delta = use_Delta
        
        # getting the global parameters
        qdPars = qdParams()

        # setting the miscellaneous directory locations
        self.local_dir = dirnames[0]
        self.scratch_dir = dirnames[1]
        self.snrnmais = dirnames[2]
        self.star_data_paths = star_data_paths
        self.datadir = f"{self.snrnmais}/data_files"
        self.outdir = f"{self.scratch_dir}/output_files"
        self.eigdir = f"{self.snrnmais}/eig_files"
        self.progdir = self.local_dir
        self.hmidata = np.loadtxt(f"{self.snrnmais}/data_files/hmi.6328.36")
        self.metadata_path = f"{metadata_path}"
        self.synthdata = f"{self.scratch_dir}/synthetic_data"

        # Frequency unit conversion factor (in Hz (cgs))
        #all quantities in cgs
        self.M_sol = 1.989e33 #gn,l = 0,200
        self.R_sol = 6.956e10 #cm
        self.B_0 = 10e5 #G
        self.OM = np.sqrt(4*np.pi*self.R_sol*self.B_0**2/self.M_sol) 
        # should be 2.096367060263202423e-05 for above numbers
        self.Teff_sun = 5780.0     # in Kelvin
        self.numax_sun = 3067.0    # in muHz
        self.g_sun = 274.0         # in SI units
        
        self.years_obs = qdPars.years_obs

        # these are the solar data
        # self.rho = np.loadtxt(f"{self.datadir}/rho.dat")
        self.r = np.loadtxt(f"{self.datadir}/r.dat")
        self.nl_all = np.loadtxt(f"{self.datadir}/nl.dat").astype('int')
        self.nl_all_list = np.loadtxt(f"{self.datadir}/nl.dat").astype('int').tolist()
        self.omega_list = np.loadtxt(f"{self.datadir}/muhz.dat") * 1e-6 / self.OM

        # getting indices for minimum and maximum r
        self.rmin = qdPars.rmin
        self.rmax = qdPars.rmax

        self.rmin_idx = self.get_idx(self.r, self.rmin)

        # removing the grid point corresponding to r=0
        # because Tsr has 1/r factor
        if self.rmin == 0:
            self.rmin_idx += 1
        self.rmax_idx = self.get_idx(self.r, self.rmax)

        self.fwindow = qdPars.fwindow

        # retaining only region between rmin and rmax
        self.r = self.mask_minmax(self.r)

        # the s array. Considering only odd s
        self.s_arr = np.arange(1, self.smax+1, 2)
        
        # the number of stars of each Stype
        self.num_startype_arr = np.load(f'{self.metadata_path}/num_startype_arr.npy')
        # the rcz of stars of each Stype
        self.rcz_startype_arr = np.load(f'{self.metadata_path}/rcz_startype_arr.npy')

    def get_idx(self, arr, val):
        return abs(arr - val).argmin()

    def mask_minmax(self, arr):
        return arr[self.rmin_idx:self.rmax_idx]
