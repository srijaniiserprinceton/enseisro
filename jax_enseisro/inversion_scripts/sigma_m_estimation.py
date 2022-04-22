import jax.numpy as jnp

def get_GT_sigd_inv(G, sigma_d):
    '''Function to compute G.T @ sigma_d_inv                                              
    for obtaining the G^{-g} later.'                                                        
    '''
    GT_sigdsq_inv= G.T @ jnp.diag(1./sigma_d**2)
    return GT_sigdsq_inv

def calc_sigma_m(inv_dicts, hess_inv):
    #-----------------finding the model covariance matrix------------------#               
    # can be shown that the model covariance matrix has the following form                  
    # C_m = G^{-g} @ C_d @ G^{-g}.T                                                          
    # G^{-g} = total_hess_inv @ G.T @ C_d_inv
    
    sigma_d = inv_dicts.data_dict['sigma_d']
    G = inv_dicts.model_dict['G']

    GT_sigdsq_inv = get_GT_sigd_inv(G, sigma_d)
    G_g_inv = hess_inv @ GT_sigdsq_inv
    C_m = G_g_inv @ jnp.diag(sigma_d**2) @ G_g_inv.T
    m_arr_err = jnp.sqrt(jnp.diag(C_m))
    return m_arr_err
