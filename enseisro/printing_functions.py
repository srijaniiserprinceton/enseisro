import numpy as np

# def format_terminal_output_onestar():                                                                                                                                             
def format_terminal_output_onestar(synthetic, inverted, error):
    """This function formats the output to be printed in the terminal                                                                                                                   
    """
    dash = '-' * 95
    col_headers = np.array(['in nHz', 'Om_in_1', 'Om_out_1', 'Om_in_3', 'Om_out_3'])
    row_headers = np.array(['Synthetic', 'Inverted', 'Error(+/-)'])

    for row_ind in range(4):
        if(row_ind == 0):
            print(dash)
            print('{:<20s}{:^20s}{:^20s}{:^20s}{:^20s}'.format(col_headers[0],\
                col_headers[1],col_headers[2],col_headers[3],col_headers[4]))
            print(dash)
        elif(row_ind == 1):
            print('{:<20s}{:^20f}{:^20f}{:^20f}{:^20f}'.format(row_headers[row_ind-1],\
                        synthetic[0],synthetic[1],synthetic[2],synthetic[3]))
        elif(row_ind == 2):
            print('{:<20s}{:^20f}{:^20f}{:^20f}{:^20f}'.format(row_headers[row_ind-1],\
                        inverted[0],inverted[1],inverted[2],inverted[3]))
        else:
            print('{:<20s}{:^20f}{:^20f}{:^20f}{:^20f}'.format(row_headers[row_ind-1],\
                        error[0],error[1],error[2],error[3]))

# }}} def format_terminal_output_onestar()   


# def format_terminal_output_ens_star():                                                                                                                                             
def format_terminal_output_ens_star(Nstars, synthetic_out, synthetic_delta, inverted_out, inverted_delta, smax, use_Delta):
    """This function formats the output to be printed in the terminal                                                                                                                   
    """
    
    if(smax == 1):
        dash = '-' * 101

        if(use_Delta):
            col_headers = np.array(['in nHz', 'Om_out_1 (Input)', 'Om_out_1 (Inverted)', \
                                    'Delta_Om_1 (Input)','Delta_Om_1 (Inverted)'])
        else: 
            col_headers = np.array(['in nHz', 'Om_in_1 (Input)', 'Om_in_1 (Inverted)', \
                                    'Om_out_1 (Input)','Om_out_1 (Inverted)'])

        row_headers = np.array([])
        
        for star_ind in range(Nstars):
            row_headers = np.append(row_headers, 'Star %i'%(star_ind+1))
            
        # printing the header row of the table
        print(dash)
        print('{:<20s}{:^20s}{:^20s}{:^20s}{:^20s}'.format(col_headers[0],\
                                            col_headers[1],col_headers[2],col_headers[3],col_headers[4]))
        print(dash)
        
        # printing the rest of the rows
        for row_ind in range(Nstars):
            print('{:<20s}{:^20f}{:^20f}{:^20f}{:^20f}'.format(row_headers[row_ind],\
            synthetic_out[row_ind],inverted_out[row_ind],synthetic_delta[0],inverted_delta[0]))


    if(smax == 3):
        dash = '-' * 182
        
        if(use_Delta):
            col_headers = np.array(['in nHz', 'Om_out_1 (Input)', 'Om_out_1 (Inverted)', \
                                    'Om_out_3 (Input)', 'Om_out_3 (Inverted)', 'Delta_Om_1 (Input)',\
                                    'Delta_Om_1 (Inverted)', 'Delta_Om_3 (Input)', 'Delta_Om_3 (inverted)'])
        else:
            col_headers = np.array(['in nHz', 'Om_in_1 (Input)', 'Om_in_1 (Inverted)', \
                                    'Om_in_3 (Input)', 'Om_in_3 (Inverted)', 'Om_out_1 (Input)',\
                                    'Om_out_1 (Inverted)', 'Om_out_3 (Input)', 'Om_out_3 (inverted)'])

        row_headers = np.array([])
        
        for star_ind in range(Nstars):
            row_headers = np.append(row_headers, 'Star %i'%(star_ind+1))
            
        # printing the header row of the table
        print(dash)
        print('{:<20s}{:^20s}{:^20s}{:^20s}{:^20s}{:^20s}{:^20s}{:^20s}{:^20s}'.format(col_headers[0],\
                                            col_headers[1],col_headers[2],col_headers[3],col_headers[4],\
                                            col_headers[5],col_headers[6],col_headers[7], col_headers[8]))
        print(dash)
        
        # printing the rest of the rows
        for row_ind in range(Nstars):
            print('{:<20s}{:^20f}{:^20f}{:^20f}{:^20f}{:^20f}{:^20f}{:^20f}{:^20f}'.format(row_headers[row_ind],\
            synthetic_out[2*row_ind],inverted_out[2*row_ind],synthetic_out[2*row_ind+1],inverted_out[2*row_ind+1],synthetic_delta[0],inverted_delta[0],synthetic_delta[1],inverted_delta[1]))

    
# }}} def format_terminal_output_ens_star()   


