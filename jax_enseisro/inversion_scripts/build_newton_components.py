import jax.numpy as jnp
import jax

#-------------data misfit function-------------#                                              
def model_fn(m_arr, inv_var_dict):
    return inv_var_dict['G'] @ (m_arr * inv_var_dict['m_arr_ref'])

def data_misfit_arr_fn(m_arr, inv_var_dict):
    model_arr = model_fn(m_arr, inv_var_dict)
    
    return (inv_var_dict['data'] - model_arr)/inv_var_dict['sigma_d']

def data_misfit_fn(m_arr, inv_var_dict):
    data_misfit_arr = data_misfit_arr_fn(m_arr, inv_var_dict)

    data_misfit = 0.5 * data_misfit_arr @ data_misfit_arr

    return data_misfit


#------------regularization function-------------#                                         
def regularization_fn(m_arr, inv_var_dict):
    return 0.0
    '''
    return inv_var_dict['mu'] * ((m_arr - inv_var_dict['m_surface']) @\
                                 (m_arr - inv_var_dict['m_surface']))
    '''


#---------------loss function-------------------#                                            
def loss_fn(m_arr, inv_var_dict):
    return data_misfit_fn(m_arr, inv_var_dict) # +\
        # regularization_fn(m_arr, inv_var_dict)
