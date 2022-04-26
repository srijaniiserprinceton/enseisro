import numpy as np

def get_Tgnu(GVARS, star_mult_arr, randomize_wrt_Sun=False):
    # number of stars in total
    Nstars  = np.sum(GVARS.num_startype_arr)
    
    Teff_arr_star = np.zeros(Nstars) + GVARS.Teff_sun
    g_arr_star = np.zeros(Nstars) + GVARS.g_sun
    numax_arr_star = np.zeros(Nstars) + GVARS.numax_sun
    
    # sigma of the normal distribution used to randomize
    T_sigma = 1000
    g_sigma = 200
    numax_sigma = 2000
    
    if(randomize_wrt_Sun):
        Teff_arr_star = np.random.uniform(GVARS.Teff_sun - T_sigma,
                                          GVARS.Teff_sun + Tsigma, Nstars)
        
        g_arr_star = np.random.uniform(GVARS.g_sun - g_sigma,
                                       GVARS.g_sun + g_sigma, Nstars)
        
        numax_arr_star = np.random.uniform(GVARS.numax_sun - numax_sigma,
                                           GVARS.numax_sun + numax_sigma, Nstars)

        
    return Teff_arr_star, g_arr_star, numax_arr_star
    
    
