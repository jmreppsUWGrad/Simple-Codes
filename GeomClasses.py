# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 23:14:27 2018

This class is a 2D planar domain for solving temp, vel or density.

Requires:
    -length and width of domain
    -number of nodes across length and width (no biasing)
    -type of problem to setup

Desired:
    -biasing ability for node creation
    -storing node spacing
    -

Features:
    -

@author: Joseph
"""
import numpy

class OneDimLine:
    def __init__(self, length, num_nodes):
        self.L=length
        self.Nx=num_nodes
        self.bias={'OneWayUp': 0, 'OneWayDown': 0, 'TwoWay': 0}
        # Setup variable arrays
        self.T=numpy.zeros(self.Nx)

class TwoDimPlanar:
    def __init__(self, length, width, node_len, node_wid, mat_prop):
        
        self.L=length
        self.W=width
        self.Nx=node_len
        self.Ny=node_wid
        self.k=mat_prop['k']
        self.Cp=mat_prop['Cp']
        self.rho=mat_prop['rho']
        
        # Boolean to determine biasing
        self.biasx=(0,0)
        self.biasy=(0,0)
        
        # Setup variable arrays
        self.T=numpy.zeros((self.Ny, self.Nx))
        self.u=numpy.zeros((self.Ny,self.Nx))
        self.v=numpy.zeros((self.Ny,self.Nx))
        self.rho=numpy.zeros((self.Ny,self.Nx))
        
    def check_var(self, var_to_check):
        if var_to_check=='T' or var_to_check=='V' or var_to_check=='rho' or var_to_check=='P':
            return ValueError('No such variable. Choose: T, V, rho, P')
        else:
            if self.Varsetup[var_to_check]==1:
                return True
            else:
                return False
    
    # Discretize domain and save dx and dy
#    def mesh(self):
    
    # Check everything before solving
#    def check_all(self):
        
