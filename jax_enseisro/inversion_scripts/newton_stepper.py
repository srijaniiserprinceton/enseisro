import time 
import jax.numpy as jnp
from jax import tree_multimap
import sys


class inv_Newton:
    def __init__(self, func_dict, inv_dicts, isIterative=False):
        #--------unwraps the inversion dictionary (static parts only)-----#
        self.loss_fn = func_dict['loss_fn']
        self.reg_fn = func_dict['reg_fn']
        self.grad_fn = func_dict['grad_fn']
        self.hess_fn = func_dict['hess_fn']
        
        #--------unwraps the inversion dictionary (static parts only)-----#    
        # creating the inverse variable dictionary
        self.inv_var_dict = {}
        self.inv_var_dict['m_arr_ref'] = inv_dicts.model_dict['model_ref']
        self.inv_var_dict['data'] = inv_dicts.data_dict['data']
        self.inv_var_dict['G'] = inv_dicts.model_dict['G']
        self.inv_var_dict['sigma_d'] = inv_dicts.data_dict['sigma_d']
        self.inv_var_dict['mu'] = inv_dicts.reg_dict['mu']
        
        # the initial model components
        self.model_init = inv_dicts.model_dict['model_init']
            
        # the loop components
        self.loss_threshold = inv_dicts.loop_dict['loss_threshold']
        self.maxiter = inv_dicts.loop_dict['maxiter']
        
        # the miscellaneous components
        self.hessinv = inv_dicts.misc_dict['hessinv']

        # if the hessian inverse has not be precomputed yet
        hess = self.hess_fn(self.model_init, self.inv_var_dict)
                            
        self.hessinv = jnp.linalg.inv(hess)
                                            

    def print_info(self, itercount, tdiff, data_misfit,
                   loss_diff, max_grads, model_misfit):
        print(f'[{itercount:3d} | ' +
              f'{tdiff:6.1f} sec ] ' +
              f'data_misfit = {data_misfit:12.5e} ' +
              f'loss-diff = {loss_diff:12.5e}; ' +
              f'max-grads = {max_grads:12.5e} ' + 
              f'model_misfit={model_misfit:12.5e}')
        
        return None


    def update(self, m_arr, grads, hess_inv):
        return tree_multimap(lambda c, g, h: c - g @ h, m_arr, grads, hess_inv)


    def run_newton(self):
        m_arr = self.model_init
        
        itercount = 0
        loss_diff = 1e25
        loss = self.loss_fn(m_arr, self.inv_var_dict)
                
        while ((abs(loss_diff) > self.loss_threshold) and
               (itercount < self.maxiter)):
            
            # start time for an iteration
            t1 = time.time()
            
            #--------------Body of a newton step---------------#
            loss_prev = loss
            
            grads = self.grad_fn(m_arr, self.inv_var_dict)
            m_arr = self.update(m_arr, grads, self.hessinv)
            loss = self.loss_fn(m_arr, self.inv_var_dict)

            
            # model_misfit = self.reg_fn(c_arr, self.model_ref, self.mu)
            model_misfit = 0.0
            data_misfit = loss -  model_misfit
            
            loss_diff = loss_prev - loss
            
            itercount += 1
            #---------------------------------------------#
            
            # end time for an iteration
            t2 = time.time()
            

            self.print_info(itercount, t2-t1, data_misfit,
                            loss_diff, abs(grads).max(), model_misfit)
                
            
        return m_arr, self.hessinv
