import jax

import build_newton_components as newton_comps
from src_iterinvPy import make_dictionaries as make_dicts

#-----------------the user-defined functions-----------------#
model_fn = newton_comps.model_fn
data_misfit_fn = newton_comps.data_misfit_fn
reg_fn = newton_comps.regularization_fn
loss_fn = newton_comps.loss_fn

#---------------------gradient and hessian-------------------#                                
grad = jax.grad(loss_fn)
hess = jax.jacfwd(jax.jacrev(loss_fn))

#-------------------jitting the functions--------------------#                                
model_fn_ = jax.jit(model_fn)
loss_fn_ = jax.jit(loss_fn)
reg_fn_ = jax.jit(reg_fn)
grad_ = jax.jit(grad)
hess_ = jax.jit(hess)

#------------------making the function dictionary-------------#
func_dict = {}
func_dict['model_fn'] = model_fn_
func_dict['loss_fn'] = loss_fn_
func_dict['reg_fn'] = reg_fn_
func_dict['grad_fn'] = grad_
func_dict['hess_fn'] = hess_

