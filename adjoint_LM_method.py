#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 15:50:32 2022

@author: georgiaacton
"""

import numpy as np

###### Calling overal optimisation

def read_from_files ():

    with open('adjoint_files/adjoint_p_values.dat') as f:
        p_value = [float(x) for x in f]
        
    with open('adjoint_files/adjoint_delta.dat') as f:
        del_value = [float(x) for x in f]
    
    with open('adjoint_files/adjoint_omega.dat') as f:
        omega_value = [[float(x) for x in line.split()] for line in f]
        
    with open('adjoint_files/adjoint_derivatives.dat') as f:
        gdt_value = [[float(x) for x in line.split()] for line in f]
    
    p_value = np.array(p_value)
    del_value = np.array(del_value)
    
    omega_value = np.array(omega_value[0])
    omega_value = complex(omega_value[0],omega_value[1])
    
    for i in range (11):
        gdt_value[i] = gdt_value[i][0]
        
    gdt_value = np.array(gdt_value)
    hess_value = np.multiply.outer(gdt_value, gdt_value)
    
    return p_value, del_value, omega_value, \
            gdt_value, hess_value

def gradient_decent (p_in, gdt) :
    
    p_in = np.array(p_in)
    gdt = np.array(gdt)    
    epsilon = 10**(-3)
    # use gradient decent to calculate next valkue of p
    p_out = p_in - epsilon * gdt
        
    return p_out

def calling_routine (): 
    
    [p_old, del_old, omega, gdt, hess] = read_from_files()   
    
    omega_store = omega
    p_new = LM_method (p_old, gdt, hess,del_old)
    
    
    return p_new. omega_store

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
p_updated, omega_previous = calling_routine()
print(p_updated)
