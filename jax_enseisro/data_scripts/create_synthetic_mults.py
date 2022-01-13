import numpy as np

def get_star_mult_arr(GVARS):
    # dictionary to store the star-wise list of multiplets                                    
    star_mult_arr = {}
    
    # the number of stars in each Star Type                                                  
    num_startype_arr = np.load(f'{GVARS.metadata}/num_startype_arr.npy')
    
    for startype_label in range(GVARS.nStype):
        num_stars_thistype = num_startype_arr[startype_label]
        
        nmin4stars = np.zeros(num_stars_thistype, dtype='int') + GVARS.nmin
        nmax4stars = np.zeros(num_stars_thistype, dtype='int') + GVARS.nmax
        
        #if slight random differences in radial orders are allowed                         
        if(GVARS.rand_mults):
            randpert_nmin = np.random.randint(-3, 4, size=num_stars_thistype)
            randpert_nmax = np.random.randint(-3, 4, size=num_stars_thistype)
            
            nmin4stars = nmin4stars + randpert_nmin
            nmax4stars = nmax4stars + randpert_nmax

        # making the list of mults for different stars of this type                      
        # will contain (star_ind, n, ell)                                                   
        mult_arr_thistype = [[np.nan, np.nan, np.nan]]

        for star_ind in range(num_stars_thistype):
            for n in range(nmin4stars[star_ind], nmax4stars[star_ind]+1):
                for ell in range(GVARS.lmin, GVARS.lmax+1):
                    mult_arr_thistype.append([star_ind, n, ell])

        mult_arr_thistype = np.asarray(mult_arr_thistype)
        # rejecting the first invalid entry                                                   
        mult_arr_thistype = mult_arr_thistype[1:].astype('int')
        
        star_mult_arr[f'{startype_label}'] = mult_arr_thistype

    return star_mult_arr
        