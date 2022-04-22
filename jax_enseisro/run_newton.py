from jax_enseisro.inversion_scripts import newton_stepper
from jax_enseisro.inversion_scripts import inv_funcs
from jax_enseisro.inversion_scripts import inv_dicts
from jax_enseisro.inversion_scripts import sigma_m_estimation

print(inv_dicts.model_dict['model_init'] / inv_dicts.model_dict['model_ref'])

# initializing the class for carrying out a single Newton inversion
invertor = newton_stepper.inv_Newton(inv_funcs.func_dict, inv_dicts)

# running one newton inversion
m_arr_fit_rel, hessinv = invertor.run_newton()

print(m_arr_fit_rel)

m_arr_fit_abs = m_arr_fit_rel * inv_dicts.model_dict['model_ref']

print(m_arr_fit_abs)

# error estimation
m_arr_err = sigma_m_estimation.calc_sigma_m(inv_dicts, hessinv)
print(m_arr_err)
