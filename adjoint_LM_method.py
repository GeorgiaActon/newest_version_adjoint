#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 15:50:32 2022

@author: georgiaacton
"""

import numpy as np
import subprocess
import f90nml
import re

###### Calling overal optimisation

def read_miller_parameters ():
    nml = f90nml.read('example.in')
    miller = [nml['millergeo_parameters']['rhoc'], \
              nml['millergeo_parameters']['shat'], \
              nml['millergeo_parameters']['qinp'], \
              nml['millergeo_parameters']['rmaj'], \
              nml['millergeo_parameters']['rgeo'], \
              nml['millergeo_parameters']['shift'], \
              nml['millergeo_parameters']['kappa'], \
              nml['millergeo_parameters']['kapprim'], \
              nml['millergeo_parameters']['tri'], \
              nml['millergeo_parameters']['triprim'], \
              nml['millergeo_parameters']['betaprim'] ]
              
    return miller

def write_miller_parameters (p):
    nml = f90nml.read('example.in')
    print(p[0])
    nml['millergeo_parameters']['rhoc'] = p[0]
    nml['millergeo_parameters']['shat'] = p[1]
    nml['millergeo_parameters']['qinp'] = p[2]
    nml['millergeo_parameters']['rmaj']	= p[3]
    nml['millergeo_parameters']['rgeo'] = p[4]
    nml['millergeo_parameters']['shift'] = p[5]
    nml['millergeo_parameters']['kappa'] = p[6]
    nml['millergeo_parameters']['kapprim'] = p[7]
    nml['millergeo_parameters']['tri'] = p[8]
    nml['millergeo_parameters']['triprim'] = p[9]
    nml['millergeo_parameters']['betaprim'] = p[10]
    
    nml.write('example.in', force=True)

    
def read_from_files ():

    # with open('adjoint_files/adjoint_p_values.dat') as f:
    #     p_value = [float(x) for x in f]
        
    with open('adjoint_files/adjoint_ginit.dat') as f:
        ginit_val = [float(x) for x in f]

    with open('adjoint_files/adjoint_gend.dat') as f:
        gfinal_val = [float(x) for x in f]
        
    with open('adjoint_files/adjoint_omega.dat') as f:
        omega_value = [[float(x) for x in line.split()] for line in f]
        
    with open('adjoint_files/adjoint_derivatives.dat') as f:
        gdt_value = [[float(x) for x in line.split()] for line in f]

    with open('adjoint_files/adjoint_final_time') as f:
        time_value = [float(x) for x in f]
        
    p_value = read_miller_parameters ()

    ginit_val = np.array(ginit_val)
    gfinal_val = np.array(gfinal_val)
    omega_value = np.array(omega_value[0])
    omega_value = complex(omega_value[0],omega_value[1])
    time_value = np.array(time_value)
    for i in range (11):
        gdt_value[i] = gdt_value[i][0]
        
    gdt_value = np.array(gdt_value)
    hess_value = np.multiply.outer(gdt_value, gdt_value)
    
    return p_value, ginit_value, gfinal_val,  omega_value,time_value, \
            gdt_value, hess_value

def gradient_decent (p_in, gdt) :
    
    p_in = np.array(p_in)
    gdt = np.array(gdt)    
    epsilon = 10**(-3)
    # use gradient decent to calculate next valkue of p
    p_out = p_in - epsilon * gdt
        
    return p_out

def calling_routine (): 
    
    [p_old, ginit, gfinal, omega, time, gdt, hess] = read_from_files()
    p_new = LM_method (p_old, gdt, hess,del_old)
    write_miller_parameters(p_new)
    
    return p_new, omega, ginit, gfinal, gdt, time

def LM_method (p_in, gdt_in, hess_in,del_old) : 
    p_in = np.array(p_in)
    gdt_in = np.array(gdt_in)
    hess_in = np.array(hess_in)  
    epsilon = 10**(-3)
    #### Matrix to invert
    A = hess_in + epsilon * np.diag(np.diag(hess_in))     
    p_out = np.linalg.solve(A, np.dot(A,p_in) - gdt_in*del_old)
        
    return p_out

#### Call this file!!!! ####
#[p_old, del_old, omega, gdt, hess] = read_from_files()

subprocess.call("./stella example.in", shell=True)
[p_updated, omega, gstart, gend, time, grad] = calling_routine()

# if (np.max(grad) < 0.1 or np.real(omega) <= 0 ) :
#         res = False
#     else :
#         res = True

# it = 0
# while(res == True):
#     subprocess.call("./stella example.in", shell=True)
    
#     [p_updated, omega, gstart, gend, grad] = calling_routine()
#     if(time[0] == time[1]) :
#         print('Omega did not converge in time limit')
#         if(gend <= gstart):
#             res = True
#         else:
#             res = False

#     else :
#         if (np.abs(np.max(grad)) < 0.01):# or f0 <= 0 ) :
#             res = False
#         else :
#             res = True
            
       
print[gstart, gend]
# with open('adjoint_files/adjoint_miller.in', 'w') as miller_out:
#     miller_out.write('adjoint_miller')

#     for i in len(p_updated) :
#         miller_out.write(miller_list(i), p_updated(i))
#         miller_out.write('\n')

#     miller_out.close()

print(p_updated)

