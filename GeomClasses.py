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
        self.L=1.0*length
        self.Nx=num_nodes
        self.x=numpy.zeros(self.Nx)
        self.dx=numpy.zeros(self.Nx-1)
        self.bias_elem={'OneWayUp': 0, 'OneWayDown': 0, 'TwoWayEnd': 0, 'TwoWayMid': 0}
        # Setup variable arrays
        self.T=numpy.zeros(self.Nx)
        self.mat_prop={'k': 5, 'Cp': 800, 'rho': 1000}
        
    def mesh(self):
        if self.bias_elem['OneWayUp']!=0:
            smallest=self.bias_elem['OneWayUp']
            self.dx=numpy.linspace(2*self.L/(self.Nx-1)-smallest,smallest,self.Nx-1)
            print 'One way biasing: smallest element at x=%2f'%self.L
        elif self.bias_elem['OneWayDown']!=0:
            smallest=self.bias_elem['OneWayDown']
            self.dx=numpy.linspace(smallest,2*self.L/(self.Nx-1)-smallest,self.Nx-1)
            print 'One way biasing: smallest element at x=0'
        elif self.bias_elem['TwoWayEnd']!=0:
            smallest=self.bias_elem['TwoWayEnd']
            self.dx[:int(self.Nx/2)]=numpy.linspace(smallest,self.L/(self.Nx-1)-smallest,(self.Nx-1)/2)
            self.dx[int(self.Nx/2):]=numpy.linspace(self.L/(self.Nx-1)-smallest,smallest,(self.Nx-1)/2)
            print 'Two way biasing: smallest elements at x=0 and %2f'%self.L
        elif self.bias_elem['TwoWayMid']!=0:
            smallest=self.bias_elem['TwoWayMid']
            self.dx[:int(self.Nx/2)]=numpy.linspace(self.L/(self.Nx-1)-smallest,smallest,(self.Nx-1)/2)
            self.dx[int(self.Nx/2):]=numpy.linspace(smallest,self.L/(self.Nx-1)-smallest,(self.Nx-1)/2)
            print 'Two way biasing: smallest elements around x=%2f'%(self.L/2)
        else:
            self.dx=numpy.linspace(0,self.L,self.Nx)
            print 'No biasing schemes specified'
        
        for i in range(self.Nx-1):
            self.x[i+1]=self.x[i]+self.dx[i]
                

class TwoDimPlanar:
    def __init__(self, length, width, node_len, node_wid, mat_prop):
        
        self.L=length
        self.W=width
        self.Nx=node_len
        self.Ny=node_wid
        self.k=mat_prop['k']
        self.Cp=mat_prop['Cp']
#        self.rho=mat_prop['rho']
        
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
        
