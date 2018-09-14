# -*- coding: utf-8 -*-
"""
Started on Wed Sep 12 21:33:59 2018

@author: Joseph

This is intended to be a 2D Navier-Stokes equation solver

Poisson solver:
    -2nd order Central difference schemes for bulk of domain

Requirements:
    -Poisson solver for bulk AND first nodes from boundary (3rd derivatives)
    -N-S solver 1st order (explicit) for time
Desired:
    -N-S solver 2nd order for time (requires 1st order for one time step)
    -

"""

# Libraries
import numpy
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D

#-------------------------------- Setup
L=3 # Length (x coordinate max)
W=1 # Width
Nx=10 # Number of nodes in x
Ny=4 # Number of nodes in y
rho=998 # Density of fluid (kg/m^3)
mu=0.001 # Dynamic viscosity of fluid (Pa s)
dt=0.01 # Time step size (s)

u=numpy.zeros((Ny, Nx))
v=numpy.zeros((Ny, Nx))
p=numpy.zeros((Ny, Nx))
dx=L/(Nx-1)
dy=W/(Ny-1)

# Boundary conditions


#-------------------------------- Solve

# Poisson equation for pressure (main)
p[2:-2,2:-2]=(dy**2*(p[3:-1,2:-2]+p[1:-3])+dx**2*(p[2:-2,3:-1]+p[2:-2,1:-3]))/(2*(dx**2+dy**2)) \
+(1/(2*(dx**2+dy**2)))*(rho*dy**2/4*(u[2:-2,3:-1]-u[2:-2,1:-3])**2+rho*dx**2/4*(v[3:-1,2:-2]-v[1:-3,2:-2])**2 \
  +rho*dx*dy/2*(u[3:-1,2:-2]-u[1:-3])*(v[2:-2,3:-1]-v[2:-2,1:-3])\
  +u[2:-2,3:-1]*(-rho*dx*dy**2/2/dt+rho*dy**2*u[2:-2,2:-2]+dx*mu+dy**2/dx*mu)\
  +u[2:-2,1:-3]*(rho*dx*dy**2/2/dt+rho*dy**2*u[2:-2,2:-2]-dx*mu-dy**2/dx*mu)\
  +u[3:-1,3:-2]*(rho*dy*dx/4*v[2:-2,3:-1]-dx*mu/2) \
  +u[3:-1,1:-3]*(-rho*dx*dy/4*v[3:-1,2:-2]-dx*mu/2) \
  +u[1:-3,3:-1]*(-rho*dy*dx/4*v[1:-3,2:-2]+dx*mu/2) \
  +u[1:-3,1:-3]*(rho*dx*dy/4*v[2:-2,1:-3]+dx*mu/2) \
  -2*rho*dy**2*u[2:-2,2:-2]**2 + dy**2*mu/2/dx*(u[:-4,2:-2]-u[4:,2:-2]) \
  +v[3:-1,3:-1]*(rho*dx*dy/4*u[2:-2,3:-1]-dy*mu/2) \
  +v[1:-3,3:-1]*(-rho*dx*dy/4*u[2:-2,3:-1]+dy*mu/2) \
  +v[3:-1,1:-3]*(-rho*dx*dy/4*u[2:-2,1:-3]-dy*mu/2) \
  +v[1:-3,1:-3]*(rho*dx*dy/4*u[2:-2,1:-3]+dy*mu/2) \
  +v[3:-1,2:-2]*(-rho*dx**2*dy/2/dt+rho*dx**2*v[2:-2,2:-2]+dy*mu+dx**2/dy*mu) \
  +v[1:-3,2:-2]*(rho*dx**2*dy/2/dt+rho*dx**2*v[2:-2,2:-2]-dy*mu-dx**2/dy*mu) \
  -2*dx**2*v[2:-2,2:-2]**2 + dx**2/2/dy*mu*(v[:-4,2:-2]-v[4:,2:-2]))
  

# Poisson equation for pressure (first nodes inside BCs)