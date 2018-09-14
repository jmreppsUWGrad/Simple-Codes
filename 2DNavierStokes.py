# -*- coding: utf-8 -*-
"""
Started on Wed Sep 12 21:33:59 2018

@author: Joseph

This is intended to be a 2D Incompressible Navier-Stokes equation solver

Poisson solver:
    -2nd order Central difference schemes for bulk of domain

Requirements:
    -Poisson solver for bulk AND first nodes from boundary (due to 3rd derivatives)
    -N-S solver 1st order (explicit) for time
Desired:
    -N-S solver 2nd order for time (requires 1st order for one time step)
    -

Function inputs:
    p: pressure array
    u,v: velocity arrays
    dxyt: array with dx, dy and dt respectively
    prop: array with fluid properties rho and mu respectively
    conv: convergence criteria
    dp_zero: array indicating which boundaries have 0 pressure gradients (normal to boundary)
    

"""

# Libraries
import numpy
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D

# Poisson solver
def ResolvePress(p, u, v, dxyt, prop, conv, dp_zero):
    rho,mu=prop
    dx,dy,dt=dxyt
    count=1
    diff=10
    error=0
    pn=p.copy()
    while (diff>conv) and (count<1000):
        
        # Poisson equation for pressure (first nodes inside boundary; small x) FIX v terms in last line
        pn[1:-1,1]=(dy**2*(p[2:,1]+p[:-2,1])+dx**2*(p[1:-1,2]+p[1:-1,0]))/(2*(dx**2+dy**2)) \
          +(1/(2*(dx**2+dy**2)))*(rho*dy**2/4*(u[1:-1,2]-u[1:-1,0])**2+rho*dx**2/4*(v[2:,1]-v[:-2,1])**2 \
          +rho*dx*dy/2*(u[2:,1]-u[:-2,1])*(v[1:-1,2]-v[1:-1,0])\
          +u[1:-1,2]*(-rho*dx*dy**2/2/dt+rho*dy**2*u[1:-1,1]+dx*mu-3*dy**2/dx*mu)\
          +u[1:-1,0]*(rho*dx*dy**2/2/dt+rho*dy**2*u[1:-1,1]-dx*mu-dy**2/dx*mu)\
          +u[2:,2]*(rho*dy*dx/4*v[1:-1,2]-dx*mu/2) \
          +u[2:,0]*(-rho*dx*dy/4*v[2:,1]-dx*mu/2) \
          +u[:-2,2]*(-rho*dy*dx/4*v[:-2,1]+dx*mu/2) \
          +u[:-2,0]*(rho*dx*dy/4*v[1:-1,0]+dx*mu/2) \
          -2*rho*dy**2*u[1:-1,1]**2 \
          +dy**2/dx*mu*u[1:-1,1]- dy**2*mu/dx*(u[1:-1,4]-u[1:-1,3]) \
          +v[2:,2]*(rho*dx*dy/4*u[1:-1,2]-dy*mu/2) \
          +v[:-2,2]*(-rho*dx*dy/4*u[1:-1,2]+dy*mu/2) \
          +v[2:,0]*(-rho*dx*dy/4*u[1:-1,0]-dy*mu/2) \
          +v[:-2,0]*(rho*dx*dy/4*u[1:-1,0]+dy*mu/2) \
          +v[2:,1]*(-rho*dx**2*dy/2/dt+rho*dx**2*v[1:-1,1]+dy*mu-dx**2/dy*mu) \
          +v[:-2,1]*(rho*dx**2*dy/2/dt+rho*dx**2*v[1:-1,1]-dy*mu-dx**2/dy*mu) \
          -2*dx**2*v[1:-1,1]**2 \
          +dx**2/dy*mu*v[1:-1,1]- dx**2/dy*mu*(v[4:,1]-v[3:,1]))
          
        # Poisson equation for pressure (first nodes inside boundary; large x)
        pn[1:-1,-2]=(dy**2*(p[2:,-2]+p[:-2,-2])+dx**2*(p[1:-1,-1]+p[1:-1,-3]))/(2*(dx**2+dy**2)) \
          +(1/(2*(dx**2+dy**2)))*(rho*dy**2/4*(u[1:-1,-1]-u[1:-1,-3])**2+rho*dx**2/4*(v[2:,-2]-v[:-2,-2])**2 \
          +rho*dx*dy/2*(u[2:,-2]-u[-3,-2])*(v[1:-1,-1]-v[1:-1,-3])\
          +u[1:-1,-1]*(-rho*dx*dy**2/2/dt+rho*dy**2*u[1:-1,-2]+dx*mu-3*dy**2/dx*mu)\
          +u[1:-1,-3]*(rho*dx*dy**2/2/dt+rho*dy**2*u[1:-1,-2]-dx*mu-dy**2/dx*mu)\
          +u[2:,-1]*(rho*dy*dx/4*v[1:-1,-1]-dx*mu/2) \
          +u[2:,-3]*(-rho*dx*dy/4*v[2:,-2]-dx*mu/2) \
          +u[:-2,-1]*(-rho*dy*dx/4*v[:-2,-2]+dx*mu/2) \
          +u[:-2,-3]*(rho*dx*dy/4*v[1:-1,-3]+dx*mu/2) \
          -2*rho*dy**2*u[1:-1,-2]**2 \
          +dy**2/dx*mu*u[1:-1,-2]- dy**2*mu/dx*(u[1:-1,4]-u[1:-1,3]) \
          +v[2:,-1]*(rho*dx*dy/4*u[1:-1,-1]-dy*mu/2) \
          +v[:-2,-1]*(-rho*dx*dy/4*u[1:-1,-1]+dy*mu/2) \
          +v[2:,-3]*(-rho*dx*dy/4*u[1:-1,-3]-dy*mu/2) \
          +v[:-2,-3]*(rho*dx*dy/4*u[1:-1,-3]+dy*mu/2) \
          +v[2:,-2]*(-rho*dx**2*dy/2/dt+rho*dx**2*v[1:-1,-2]+dy*mu-dx**2/dy*mu) \
          +v[:-2,-2]*(rho*dx**2*dy/2/dt+rho*dx**2*v[1:-1,-2]-dy*mu-dx**2/dy*mu) \
          -2*dx**2*v[1:-1,-2]**2 \
          +dx**2/dy*mu*v[1:-1,-2]- dx**2/dy*mu*(v[4,1:-1]-v[3,1:-1]))  
        # Poisson equation for pressure (first nodes inside boundary; small y)
          
        # Poisson equation for pressure (first nodes inside boundary; large y)
        
        # Poisson equation for pressure (main)
        pn[2:-2,2:-2]=(dx**2*(p[3:-1,2:-2]+p[1:-3,2:-2])+dy**2*(p[2:-2,3:-1]+p[2:-2,1:-3]))/(2*(dx**2+dy**2)) \
          +(1/(2*(dx**2+dy**2)))*(rho*dy**2/4*(u[2:-2,3:-1]-u[2:-2,1:-3])**2+rho*dx**2/4*(v[3:-1,2:-2]-v[1:-3,2:-2])**2 \
          +rho*dx*dy/2*(u[3:-1,2:-2]-u[1:-3,2:-2])*(v[2:-2,3:-1]-v[2:-2,1:-3])\
          +u[2:-2,3:-1]*(-rho*dx*dy**2/2/dt+rho*dy**2*u[2:-2,2:-2]+dx*mu+dy**2/dx*mu)\
          +u[2:-2,1:-3]*(rho*dx*dy**2/2/dt+rho*dy**2*u[2:-2,2:-2]-dx*mu-dy**2/dx*mu)\
          +u[3:-1,3:-1]*(rho*dy*dx/4*v[2:-2,3:-1]-dx*mu/2) \
          +u[3:-1,1:-3]*(-rho*dx*dy/4*v[3:-1,2:-2]-dx*mu/2) \
          +u[1:-3,3:-1]*(-rho*dy*dx/4*v[1:-3,2:-2]+dx*mu/2) \
          +u[1:-3,1:-3]*(rho*dx*dy/4*v[2:-2,1:-3]+dx*mu/2) \
          -2*rho*dy**2*u[2:-2,2:-2]**2 + dy**2*mu/2/dx*(u[2:-2,:-4]-u[2:-2, 4:]) \
          +v[3:-1,3:-1]*(rho*dx*dy/4*u[2:-2,3:-1]-dy*mu/2) \
          +v[1:-3,3:-1]*(-rho*dx*dy/4*u[2:-2,3:-1]+dy*mu/2) \
          +v[3:-1,1:-3]*(-rho*dx*dy/4*u[2:-2,1:-3]-dy*mu/2) \
          +v[1:-3,1:-3]*(rho*dx*dy/4*u[2:-2,1:-3]+dy*mu/2) \
          +v[3:-1,2:-2]*(-rho*dx**2*dy/2/dt+rho*dx**2*v[2:-2,2:-2]+dy*mu+dx**2/dy*mu) \
          +v[1:-3,2:-2]*(rho*dx**2*dy/2/dt+rho*dx**2*v[2:-2,2:-2]-dy*mu-dx**2/dy*mu) \
          -2*dx**2*v[2:-2,2:-2]**2 + dx**2/2/dy*mu*(v[:-4,2:-2]-v[4:,2:-2]))
        
          # Convergence check
        diff=numpy.sum(numpy.abs(p[:]-pn[:]))/numpy.sum(numpy.abs(p[:]))
        p=pn.copy()
        count=count+1
    if count==1000:
        print 'Convergence problem for pressure distribution'
        error=1
    return p,error


#-------------------------------- Setup
L=3.0 # Length (x coordinate max)
W=1.0 # Width
Nx=10 # Number of nodes in x
Ny=4 # Number of nodes in y
rho=998 # Density of fluid (kg/m^3)
mu=0.001 # Dynamic viscosity of fluid (Pa s)
dt=0.01 # Time step size (s)
Nt=100 # Number of time steps

u=numpy.zeros((Ny, Nx))
v=numpy.zeros((Ny, Nx))
p=numpy.zeros((Ny, Nx))
dx=L/(Nx-1)
dy=W/(Ny-1)
nu=mu/rho

# Convergence
conv=0.001 # convergence criteria

# Boundary conditions


#-------------------------------- Solve

for i in range(Nt):
    
    # Solve pressure field
    p,error=ResolvePress(p, u, v, (dx, dy, dt), (rho, mu), conv)
    
    # Solve momentum equations (explicit, first order for time)
    un=u.copy()
    un[1:-1,1:-1]=dt/(2*rho*dx)*(p[1:-1,:-2]+p[1:-1,2:]) \
      +u[1:-1,2:]*(dt*nu/dx**2-dt/2/dx*u[1:-1,1:-1]) \
      +u[1:-1,:-2]*(dt*nu/dx**2+dt/2/dx*u[1:-1,1:-1]) \
      +u[2:,1:-1]*(dt*nu/dy**2-dt/2/dy*v[1:-1,1:-1]) \
      +u[:-2,1:-1]*(dt*nu/dy**2+dt/2/dy*v[1:-1,1:-1]) \
      +u[1:-1,1:-1]*(1-2*nu*dt*(1/dx**2+1/dy**2))
    
    vn=v.copy()
    vn[1:-1,1:-1]=dt/(2*rho*dy)*(p[:-2,1:-1]-p[2:,1:-1]) \
      +v[1:-1,2:]*(dt*nu/dx**2-dt/2/dx*u[1:-1,1:-1]) \
      +v[1:-1,:-2]*(dt*nu/dx**2+dt/2/dx*v[1:-1,1:-1]) \
      +v[2:,1:-1]*(dt*nu/dy**2-dt/2/dy*v[1:-1,1:-1]) \
      +v[:-2,1:-1]*(dt*nu/dy**2+dt/2/dy*v[1:-1,1:-1]) \
      +v[1:-1,1:-1]*(1-2*dt*nu*(1/dx**2+1/dy**2))