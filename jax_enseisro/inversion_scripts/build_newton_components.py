import jax.numpy as jnp
import jax

#-------------data misfit function-------------#                                              
def model_fn(m_arr, G):
    return m_arr @ G

def data_misfit_fn(m_arr, data, G, C_d):
    data_res = data - m_arr @ G
    C_d_inv = jnp.linalg.inv(C_d)

    data_misfit = data_res @ C_d_inv @ data_res

    return data_misfit

#------------regularization function-------------#                                         
def regularization_fn(m_arr, m_ref_arr, mu):
    return mu * (m_arr - m_ref_arr)**2

#---------------loss function-------------------#                                            
def loss_fn(m_arr, data, G, C_d, m_ref_arr, mu):
    return data_misfit_fn(m_arr, data, G, C_d) # +\
        # model_misfit_fn(m_arr, m_ref_arr, mu)

def get_newton_components():
    # defining the gradient                                                               
    grad = jax.grad(loss_fn)

    # defining the hessian                                                             
    hess = jax.jacfwd(jax.jacrev(loss_fn))

    loss_fn_ = jax.jit(loss_fn)
    model_misfit_fn_ = jax.jit(model_misfit_fn)
    grad_ = jax.jit(grad)
    hess_ = jax.jit(hess)

    return loss_fn_, model_misfit_fn_, grad_, hess_
