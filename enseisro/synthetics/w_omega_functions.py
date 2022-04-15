import jax.numpy as np
import numpy as np
from scipy.special import legendre

NAX = np.newaxis

def adjust_ens_dim(wsr):
    '''# checking the dimensions of wsr. If ndim=2 then expanding by 1 dim                   
    this allows for a generic handling if the ensemble form (s x r x N)                       
    is passed
    '''
    if(wsr.ndim == 2):
        lens, lenr = np.shape(wsr)
        wsr = np.reshape(wsr, (lens, lenr, 1))
        lenN = 1
    else:
        lens, lenr, lenN = np.shape(wsr)

    return lens, lenr, lenN, wsr

def w_2_Omega(GVAR, wsr):
    '''This function converts w_s(r) to \Omega_s(r)
    as can be found in Eqn.~(9) in Vorontsov 2011
    '''
    # readjusting ensemble dimension to take care of both
    # ensemble and non-ensemble case
    lens, lenr, lenN, wsr = adjust_ens_dim(wsr)

    # creating the Omega_s(r) matrix
    Omegasr = np.zeros_like(wsr)

    for s_ind in range(lens):
        s = 2 * s_ind + 1
        
        # Shape: r
        conv_factor = np.sqrt((2 * s + 1)/(4*np.pi)) / GVAR.r

        Omegasr[s_ind,:,:] = conv_factor[:,NAX] * wsr[s_ind,:,:]

    
    # if its just one star and not an ensemble
    if(lenN == 1):
        Omegasr = Omegasr[:,:,0]  # getting rid of extra dimension

    return Omegasr

def Omega_2_w(GVAR, Omega_sr):
    '''This function converts \Omega_s(r) to w_s(r)                                           
    as can be found in Eqn.~(9) in Vorontsov 2011                                             
    '''
    # readjusting ensemble dimension to take care of both                                     
    # ensemble and non-ensemble case                                                          
    lens, lenr, lenN, Omega_sr = adjust_ens_dim(Omega_sr)

    # creating the w_s(r) matrix                                                          
    wsr = np.zeros_like(Omega_sr)

    for s_ind in range(lens):
        s = 2 * s_ind + 1

        # Shape: r                                                                            
        conv_factor = np.sqrt((4 * np.pi) / (2 * s + 1)) * GVAR.r

        wsr[s_ind,:,:] = conv_factor[:,NAX] * Omega_sr[s_ind,:,:]


    # if its just one star and not an ensemble                                                
    if(lenN == 1):
        wsr = wsr[:,:,0]  # getting rid of extra dimension                            

    return wsr

def Omega_s_to_Omega_theta(Omegasr, Ntheta=100):
    '''This function converts Omega_s(r) to Omega(r,theta).
    This uses the basis decomposition as in Eqn,~(10) of Vorontsov11
    '''

    lens, lenr, lenN, Omegasr = adjust_ens_dim(Omegasr)

    # theta grid
    theta = np.linspace(0,np.pi/2,Ntheta)

    # cos(theta) grid that goes into legendre polynomials
    costheta = np.cos(theta)

    # Omega(r,theta) profile. (r x theta x N)
    Omega_r_theta = np.zeros((lenr, Ntheta, lenN))

    # carrying out the sum over angular degree s
    for s_ind in range(lens):
        s = 2 * s_ind + 1    # we want odd s only. s = 1, 3, ...
        
        # creating the legendre polynomial. P_s(x)
        leg_poly = legendre(s)
        # derivative with respect to its argument dP_s(x)/dx
        leg_poly_deriv = leg_poly.deriv()

        Omega_r_theta += Omegasr[s_ind,:,NAX,:] * leg_poly_deriv(costheta)[NAX,:,NAX]

    # if its just one star and not an ensemble                                                   
    if(lenN == 1):
        Omega_r_theta = Omega_r_theta[:,:,0]  # getting rid of extra dimension

    return Omega_r_theta, theta
        
        
        
        
    
