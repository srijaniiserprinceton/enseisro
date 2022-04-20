import numpy as np

def make_model_params(Omega_step, num_model_params, GVARS):
    # for now we use Omega_step of shape (3 x r)
    Omega_1_in = Omega_step[0,0]
    
    # number of s components fitting for
    len_s = len(GVARS.s_arr)
    
    # making the array of Delta_Omega_s
    Delta_Omega_s_arr = np.zeros(len_s)
    
    # this is only for s=1 since interior step is not zero
    Delta_Omega_s_arr[0] = Omega_step[0,-1] - Omega_step[0,0]
    
    # for s=3,5,... where the Omega_step in the interior is zero
    for sind, s in enumerate(GVARS.s_arr[1:]):
        Delta_Omega_s_arr[sind+1] = Omega_step[sind, -1]
        

    model_params_G = np.zeros(num_model_params)
    model_params_G[0:-len_s] = Omega_1_in
    model_params_G[-len_s:] = Delta_Omega_s_arr
    
    return model_params_G
