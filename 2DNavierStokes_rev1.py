# -*- coding: utf-8 -*-
"""
Started on Wed Sep 12 21:33:59 2018

@author: Joseph

This is a 2D Incompressible, unsteady Navier-Stokes equation solver

FEATURES:
    -2nd order central difference schemes for all spatial derivatives
    -1st order time, explicit formulation
    -able to compute periodic and 0 gradient BCs for pressure
    -able to compute periodic BC for velocities
    -Constant pressure gradient for Poiseuille flow possible (Cannot vary spatially)
    -pressure clipping

Poisson solver:
    -2nd order Central difference schemes for bulk of domain
    -1st order forward or backwards differences for nodes 1 away from boundaries

THIS REVISION (rev1 from 2DNavierStokes.py):
    -Rewrote pressure solver without collecting terms
    -Manally rewrote each pressure calculation step
    -Confirmed it replicates Step 11 in 12 steps to N-S
    -After adding source terms, replicate parabolic velocity profile 
    as seen in step 12 of 12 steps to N-S, but not to same velocity values

Function inputs:
    p: pressure array
    u,v: velocity arrays
    dxyt: array with dx, dy and dt respectively
    prop: array with fluid properties rho and mu respectively
    conv: convergence criteria
    dp_zero: array indicating pressre BCs
            2-periodic, 1-zero pressure gradient, 0-regular BC 
            e.g. 1010-zero pressure gradient at smallest x and y
            2010-periodic BC in x (implied both sides), 0 pressure grad on small y

"""

# Libraries
import numpy
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D

# Poisson solver
def ResolvePress(p, u, v, dxyt, prop, conv, YesRes, dp_zero):
    rho,mu=prop
    dx,dy,dt=dxyt
    count=1
    diff=10
    error=0
    pn=p.copy()
    while (diff>conv) and (count<1000):
        
        # Poisson equation for pressure (first nodes inside boundary; small x)
        st=1
        en=-3
        sin=1
        pn[st:en,sin]=dy**2/(2*dx**2+2*dy**2)*(p[st:en,sin+1]+p[st:en,sin-1])+dx**2/(2*dx**2+2*dy**2)*(p[st+1:en+1,sin]+p[st-1:en-1,sin]) \
        -dx**2*dy**2/(2*dx**2+2*dy**2)*(rho*(1/dt*((u[st:en,sin+1]-u[st:en,sin-1])/2/dx+(v[st+1:en+1,sin]-v[st-1:en-1,sin])/2/dy) \
        -u[st:en,sin]*(u[st:en,sin-1]-2*u[st:en,sin]+u[st:en,sin+1])/dx**2-((u[st:en,sin+1]-u[st:en,sin-1])/2/dx)**2\
          -((v[st+1:en+1,sin]-v[st-1:en-1,sin])/2/dy)**2-v[st:en,sin]*(v[st-1:en-1,sin]-2*v[st:en,sin]+v[st+1:en+1,sin])/dy**2 \
          -(u[st:en,sin+1]*(v[st+1:en+1,sin+1]-v[st-1:en-1,sin+1])/4/dy/dx+v[st:en,sin+1]*(u[st+1:en+1,sin+1]-u[st-1:en-1,sin+1])/4/dy/dx) \
          +u[st:en,sin-1]*(v[st+1:en+1,sin-1]-v[st-1:en-1,sin-1])/4/dy/dx+v[st:en,sin-1]*(u[st+1:en+1,sin-1]-u[st-1:en-1,sin-1])/4/dy/dx \
          -2*(u[st+1:en+1,sin]-u[st-1:en-1,sin])*(v[st:en,sin+1]-v[st:en,sin-1])/4/dy/dx)\
          +mu*((u[st+1:en+1,sin+1]-2*u[st:en,sin+1]+u[st-1:en-1,sin+1])/2/dx/dy**2-(u[st+1:en+1,sin-1]-2*u[st:en,sin-1]+u[st-1:en-1,sin-1])/2/dx/dy**2 \
       +(v[st+1:en+1,sin+1]-2*v[st+1:en+1,sin]+v[st+1:en+1,sin-1])/2/dy/dx**2-(v[st-1:en-1,sin+1]-2*v[st-1:en-1,sin]+v[st-1:en-1,sin-1])/2/dy/dx**2 \
       +(u[st:en,sin+3]-3*u[st:en,sin+2]+3*u[st:en,sin+1]-u[st:en,sin])/dx**3\
       +(v[4:,sin]-3*v[st+2:en+2,sin]+3*v[st+1:en+1,sin]-v[st:en,sin])/dy**3))
        
        st=-4
        en=-1
        pn[st:en,sin]=dy**2/(2*dx**2+2*dy**2)*(p[st:en,sin+1]+p[st:en,sin-1])+dx**2/(2*dx**2+2*dy**2)*(p[st+1:,sin]+p[st-1:en-1,sin]) \
        -dx**2*dy**2/(2*dx**2+2*dy**2)*(rho*(1/dt*((u[st:en,sin+1]-u[st:en,sin-1])/2/dx+(v[st+1:,sin]-v[st-1:en-1,sin])/2/dy) \
        -u[st:en,sin]*(u[st:en,sin-1]-2*u[st:en,sin]+u[st:en,sin+1])/dx**2-((u[st:en,sin+1]-u[st:en,sin-1])/2/dx)**2\
          -((v[st+1:,sin]-v[st-1:en-1,sin])/2/dy)**2-v[st:en,sin]*(v[st-1:en-1,sin]-2*v[st:en,sin]+v[st+1:,sin])/dy**2 \
          -(u[st:en,sin+1]*(v[st+1:,sin+1]-v[st-1:en-1,sin+1])/4/dy/dx+v[st:en,sin+1]*(u[st+1:,sin+1]-u[st-1:en-1,sin+1])/4/dy/dx) \
          +u[st:en,sin-1]*(v[st+1:,sin-1]-v[st-1:en-1,sin-1])/4/dy/dx+v[st:en,sin-1]*(u[st+1:,sin-1]-u[st-1:en-1,sin-1])/4/dy/dx \
          -2*(u[st+1:,sin]-u[st-1:en-1,sin])*(v[st:en,sin+1]-v[st:en,sin-1])/4/dy/dx)\
          +mu*((u[st+1:,sin+1]-2*u[st:en,sin+1]+u[st-1:en-1,sin+1])/2/dx/dy**2-(u[st+1:,sin-1]-2*u[st:en,sin-1]+u[st-1:en-1,sin-1])/2/dx/dy**2 \
       +(v[st+1:,sin+1]-2*v[st+1:,sin]+v[st+1:,sin-1])/2/dy/dx**2-(v[st-1:en-1,sin+1]-2*v[st-1:en-1,sin]+v[st-1:en-1,sin-1])/2/dy/dx**2 \
       +(u[st:en,sin+3]-3*u[st:en,sin+2]+3*u[st:en,sin+1]-u[st:en,sin])/dx**3\
       -(v[st-3:en-3,sin]-3*v[st-2:en-2,sin]+3*v[st-1:en-1,sin]-v[st:en,sin])/dy**3))
        
        # Poisson equation for pressure (first nodes inside boundary; large y)
        st=2
        en=-3
        sin=-2
        pn[sin,st:en]=dy**2/(2*dx**2+2*dy**2)*(p[sin,st+1:en+1]+p[sin,st-1:en-1])+dx**2/(2*dx**2+2*dy**2)*(p[sin+1,st:en]+p[sin-1,st:en]) \
        -dx**2*dy**2/(2*dx**2+2*dy**2)*(rho*(1/dt*((u[sin,st+1:en+1]-u[sin,st-1:en-1])/2/dx+(v[sin+1,st:en]-v[sin-1,st:en])/2/dy) \
        -u[sin,st:en]*(u[sin,st-1:en-1]-2*u[sin,st:en]+u[sin,st+1:en+1])/dx**2-((u[sin,st+1:en+1]-u[sin,st-1:en-1])/2/dx)**2\
          -((v[sin+1,st:en]-v[sin-1,st:en])/2/dy)**2-v[sin,st:en]*(v[sin-1,st:en]-2*v[sin,st:en]+v[sin+1,st:en])/dy**2 \
          -(u[sin,st+1:en+1]*(v[sin+1,st+1:en+1]-v[sin-1,st+1:en+1])/4/dy/dx+v[sin,st+1:en+1]*(u[sin+1,st+1:en+1]-u[sin-1,st+1:en+1])/4/dy/dx) \
          +u[sin,st-1:en-1]*(v[sin+1,st-1:en-1]-v[sin-1,st-1:en-1])/4/dy/dx+v[sin,st-1:en-1]*(u[sin+1,st-1:en-1]-u[sin-1,st-1:en-1])/4/dy/dx \
          -2*(u[sin+1,st:en]-u[sin-1,st:en])*(v[sin,st+1:en+1]-v[sin,st-1:en-1])/4/dy/dx)\
          +mu*((u[sin+1,st+1:en+1]-2*u[sin,st+1:en+1]+u[sin-1,st+1:en+1])/2/dx/dy**2-(u[sin+1,st-1:en-1]-2*u[sin,st-1:en-1]+u[sin-1,st-1:en-1])/2/dx/dy**2 \
           +(v[sin+1,st+1:en+1]-2*v[sin+1,st:en]+v[sin+1,st-1:en-1])/2/dy/dx**2-(v[sin-1,st+1:en+1]-2*v[sin-1,st:en]+v[sin-1,st-1:en-1])/2/dy/dx**2 \
           +(u[sin,5:]-3*u[sin,st+2:en+2]+3*u[sin,st+1:en+1]-u[sin,st:en])/dx**3\
           -(v[sin-3,st:en]-3*v[sin-2,st:en]+3*v[sin-1,st:en]-v[sin,st:en])/dy**3))
                
        st=-4
        en=-1
        pn[sin,st:en]=dy**2/(2*dx**2+2*dy**2)*(p[sin,st+1:]+p[sin,st-1:en-1])+dx**2/(2*dx**2+2*dy**2)*(p[sin+1,st:en]+p[sin-1,st:en]) \
        -dx**2*dy**2/(2*dx**2+2*dy**2)*(rho*(1/dt*((u[sin,st+1:]-u[sin,st-1:en-1])/2/dx+(v[sin+1,st:en]-v[sin-1,st:en])/2/dy) \
        -u[sin,st:en]*(u[sin,st-1:en-1]-2*u[sin,st:en]+u[sin,st+1:])/dx**2-((u[sin,st+1:]-u[sin,st-1:en-1])/2/dx)**2\
          -((v[sin+1,st:en]-v[sin-1,st:en])/2/dy)**2-v[sin,st:en]*(v[sin-1,st:en]-2*v[sin,st:en]+v[sin+1,st:en])/dy**2 \
          -(u[sin,st+1:]*(v[sin+1,st+1:]-v[sin-1,st+1:])/4/dy/dx+v[sin,st+1:]*(u[sin+1,st+1:]-u[sin-1,st+1:])/4/dy/dx) \
          +u[sin,st-1:en-1]*(v[sin+1,st-1:en-1]-v[sin-1,st-1:en-1])/4/dy/dx+v[sin,st-1:en-1]*(u[sin+1,st-1:en-1]-u[sin-1,st-1:en-1])/4/dy/dx \
          -2*(u[sin+1,st:en]-u[sin-1,st:en])*(v[sin,st+1:]-v[sin,st-1:en-1])/4/dy/dx)\
          +mu*((u[sin+1,st+1:]-2*u[sin,st+1:]+u[sin-1,st+1:])/2/dx/dy**2-(u[sin+1,st-1:en-1]-2*u[sin,st-1:en-1]+u[sin-1,st-1:en-1])/2/dx/dy**2 \
           +(v[sin+1,st+1:]-2*v[sin+1,st:en]+v[sin+1,st-1:en-1])/2/dy/dx**2-(v[sin-1,st+1:]-2*v[sin-1,st:en]+v[sin-1,st-1:en-1])/2/dy/dx**2 \
           -(u[sin,st-3:en-3]-3*u[sin,st-2:en-2]+3*u[sin,st-1:en-1]-u[sin,st:en])/dx**3\
           -(v[sin-3,st:en]-3*v[sin-2,st:en]+3*v[sin-1,st:en]-v[sin,st:en])/dy**3))
        
        # Poisson equation for pressure (first nodes inside boundary; large x)
        st=3
        en=-2
        sin=-2
        pn[st:en,sin]=dy**2/(2*dx**2+2*dy**2)*(p[st:en,sin+1]+p[st:en,sin-1])+dx**2/(2*dx**2+2*dy**2)*(p[st+1:en+1,sin]+p[st-1:en-1,sin]) \
        -dx**2*dy**2/(2*dx**2+2*dy**2)*(rho*(1/dt*((u[st:en,sin+1]-u[st:en,sin-1])/2/dx+(v[st+1:en+1,sin]-v[st-1:en-1,sin])/2/dy) \
        -u[st:en,sin]*(u[st:en,sin-1]-2*u[st:en,sin]+u[st:en,sin+1])/dx**2-((u[st:en,sin+1]-u[st:en,sin-1])/2/dx)**2\
          -((v[st+1:en+1,sin]-v[st-1:en-1,sin])/2/dy)**2-v[st:en,sin]*(v[st-1:en-1,sin]-2*v[st:en,sin]+v[st+1:en+1,sin])/dy**2 \
          -(u[st:en,sin+1]*(v[st+1:en+1,sin+1]-v[st-1:en-1,sin+1])/4/dy/dx+v[st:en,sin+1]*(u[st+1:en+1,sin+1]-u[st-1:en-1,sin+1])/4/dy/dx) \
          +u[st:en,sin-1]*(v[st+1:en+1,sin-1]-v[st-1:en-1,sin-1])/4/dy/dx+v[st:en,sin-1]*(u[st+1:en+1,sin-1]-u[st-1:en-1,sin-1])/4/dy/dx \
          -2*(u[st+1:en+1,sin]-u[st-1:en-1,sin])*(v[st:en,sin+1]-v[st:en,sin-1])/4/dy/dx)\
          +mu*((u[st+1:en+1,sin+1]-2*u[st:en,sin+1]+u[st-1:en-1,sin+1])/2/dx/dy**2-(u[st+1:en+1,sin-1]-2*u[st:en,sin-1]+u[st-1:en-1,sin-1])/2/dx/dy**2 \
       +(v[st+1:en+1,sin+1]-2*v[st+1:en+1,sin]+v[st+1:en+1,sin-1])/2/dy/dx**2-(v[st-1:en-1,sin+1]-2*v[st-1:en-1,sin]+v[st-1:en-1,sin-1])/2/dy/dx**2 \
       -(u[st:en,sin-3]-3*u[st:en,sin-2]+3*u[st:en,sin-1]-u[st:en,sin])/dx**3\
       -(v[st-3:en-3,sin]-3*v[st-2:en-2,sin]+3*v[st-1:en-1,sin]-v[st:en,sin])/dy**3))
        
        st=1
        en=4
        pn[st:en,sin]=dy**2/(2*dx**2+2*dy**2)*(p[st:en,sin+1]+p[st:en,sin-1])+dx**2/(2*dx**2+2*dy**2)*(p[st+1:en+1,sin]+p[st-1:en-1,sin]) \
        -dx**2*dy**2/(2*dx**2+2*dy**2)*(rho*(1/dt*((u[st:en,sin+1]-u[st:en,sin-1])/2/dx+(v[st+1:en+1,sin]-v[st-1:en-1,sin])/2/dy) \
        -u[st:en,sin]*(u[st:en,sin-1]-2*u[st:en,sin]+u[st:en,sin+1])/dx**2-((u[st:en,sin+1]-u[st:en,sin-1])/2/dx)**2\
          -((v[st+1:en+1,sin]-v[st-1:en-1,sin])/2/dy)**2-v[st:en,sin]*(v[st-1:en-1,sin]-2*v[st:en,sin]+v[st+1:en+1,sin])/dy**2 \
          -(u[st:en,sin+1]*(v[st+1:en+1,sin+1]-v[st-1:en-1,sin+1])/4/dy/dx+v[st:en,sin+1]*(u[st+1:en+1,sin+1]-u[st-1:en-1,sin+1])/4/dy/dx) \
          +u[st:en,sin-1]*(v[st+1:en+1,sin-1]-v[st-1:en-1,sin-1])/4/dy/dx+v[st:en,sin-1]*(u[st+1:en+1,sin-1]-u[st-1:en-1,sin-1])/4/dy/dx \
          -2*(u[st+1:en+1,sin]-u[st-1:en-1,sin])*(v[st:en,sin+1]-v[st:en,sin-1])/4/dy/dx)\
          +mu*((u[st+1:en+1,sin+1]-2*u[st:en,sin+1]+u[st-1:en-1,sin+1])/2/dx/dy**2-(u[st+1:en+1,sin-1]-2*u[st:en,sin-1]+u[st-1:en-1,sin-1])/2/dx/dy**2 \
       +(v[st+1:en+1,sin+1]-2*v[st+1:en+1,sin]+v[st+1:en+1,sin-1])/2/dy/dx**2-(v[st-1:en-1,sin+1]-2*v[st-1:en-1,sin]+v[st-1:en-1,sin-1])/2/dy/dx**2 \
       -(u[st:en,sin-3]-3*u[st:en,sin-2]+3*u[st:en,sin-1]-u[st:en,sin])/dx**3\
       +(v[st+3:en+3,sin]-3*v[st+2:en+2,sin]+3*v[st+1:en+1,sin]-v[st:en,sin])/dy**3))
        
        # Poisson equation for pressure (first nodes inside boundary; small y)
        st=3
        en=-2
        sin=1
        pn[sin,st:en]=dy**2/(2*dx**2+2*dy**2)*(p[sin,st+1:en+1]+p[sin,st-1:en-1])+dx**2/(2*dx**2+2*dy**2)*(p[sin+1,st:en]+p[sin-1,st:en]) \
        -dx**2*dy**2/(2*dx**2+2*dy**2)*(rho*(1/dt*((u[sin,st+1:en+1]-u[sin,st-1:en-1])/2/dx+(v[sin+1,st:en]-v[sin-1,st:en])/2/dy) \
        -u[sin,st:en]*(u[sin,st-1:en-1]-2*u[sin,st:en]+u[sin,st+1:en+1])/dx**2-((u[sin,st+1:en+1]-u[sin,st-1:en-1])/2/dx)**2\
          -((v[sin+1,st:en]-v[sin-1,st:en])/2/dy)**2-v[sin,st:en]*(v[sin-1,st:en]-2*v[sin,st:en]+v[sin+1,st:en])/dy**2 \
          -(u[sin,st+1:en+1]*(v[sin+1,st+1:en+1]-v[sin-1,st+1:en+1])/4/dy/dx+v[sin,st+1:en+1]*(u[sin+1,st+1:en+1]-u[sin-1,st+1:en+1])/4/dy/dx) \
          +u[sin,st-1:en-1]*(v[sin+1,st-1:en-1]-v[sin-1,st-1:en-1])/4/dy/dx+v[sin,st-1:en-1]*(u[sin+1,st-1:en-1]-u[sin-1,st-1:en-1])/4/dy/dx \
          -2*(u[sin+1,st:en]-u[sin-1,st:en])*(v[sin,st+1:en+1]-v[sin,st-1:en-1])/4/dy/dx)\
          +mu*((u[sin+1,st+1:en+1]-2*u[sin,st+1:en+1]+u[sin-1,st+1:en+1])/2/dx/dy**2-(u[sin+1,st-1:en-1]-2*u[sin,st-1:en-1]+u[sin-1,st-1:en-1])/2/dx/dy**2 \
           +(v[sin+1,st+1:en+1]-2*v[sin+1,st:en]+v[sin+1,st-1:en-1])/2/dy/dx**2-(v[sin-1,st+1:en+1]-2*v[sin-1,st:en]+v[sin-1,st-1:en-1])/2/dy/dx**2 \
           -(u[sin,st-3:en-3]-3*u[sin,st-2:en-2]+3*u[sin,st-1:en-1]-u[sin,st:en])/dx**3\
           +(v[sin+3,st:en]-3*v[sin+2,st:en]+3*v[sin+1,st:en]-v[sin,st:en])/dy**3))
        
        st=2
        en=3
        pn[sin,st:en]=dy**2/(2*dx**2+2*dy**2)*(p[sin,st+1:en+1]+p[sin,st-1:en-1])+dx**2/(2*dx**2+2*dy**2)*(p[sin+1,st:en]+p[sin-1,st:en]) \
        -dx**2*dy**2/(2*dx**2+2*dy**2)*(rho*(1/dt*((u[sin,st+1:en+1]-u[sin,st-1:en-1])/2/dx+(v[sin+1,st:en]-v[sin-1,st:en])/2/dy) \
        -u[sin,st:en]*(u[sin,st-1:en-1]-2*u[sin,st:en]+u[sin,st+1:en+1])/dx**2-((u[sin,st+1:en+1]-u[sin,st-1:en-1])/2/dx)**2\
          -((v[sin+1,st:en]-v[sin-1,st:en])/2/dy)**2-v[sin,st:en]*(v[sin-1,st:en]-2*v[sin,st:en]+v[sin+1,st:en])/dy**2 \
          -(u[sin,st+1:en+1]*(v[sin+1,st+1:en+1]-v[sin-1,st+1:en+1])/4/dy/dx+v[sin,st+1:en+1]*(u[sin+1,st+1:en+1]-u[sin-1,st+1:en+1])/4/dy/dx) \
          +u[sin,st-1:en-1]*(v[sin+1,st-1:en-1]-v[sin-1,st-1:en-1])/4/dy/dx+v[sin,st-1:en-1]*(u[sin+1,st-1:en-1]-u[sin-1,st-1:en-1])/4/dy/dx \
          -2*(u[sin+1,st:en]-u[sin-1,st:en])*(v[sin,st+1:en+1]-v[sin,st-1:en-1])/4/dy/dx)\
          +mu*((u[sin+1,st+1:en+1]-2*u[sin,st+1:en+1]+u[sin-1,st+1:en+1])/2/dx/dy**2-(u[sin+1,st-1:en-1]-2*u[sin,st-1:en-1]+u[sin-1,st-1:en-1])/2/dx/dy**2 \
           +(v[sin+1,st+1:en+1]-2*v[sin+1,st:en]+v[sin+1,st-1:en-1])/2/dy/dx**2-(v[sin-1,st+1:en+1]-2*v[sin-1,st:en]+v[sin-1,st-1:en-1])/2/dy/dx**2 \
           +(u[sin,st+3:en+3]-3*u[sin,st+2:en+2]+3*u[sin,st+1:en+1]-u[sin,st:en])/dx**3\
           +(v[sin+3,st:en]-3*v[sin+2,st:en]+3*v[sin+1,st:en]-v[sin,st:en])/dy**3))
    
        # Poisson equation for pressure (main)-redo, no collection of terms
        pn[2:-2,2:-2]=dy**2/(2*dx**2+2*dy**2)*(p[2:-2,3:-1]+p[2:-2,1:-3])+dx**2/(2*dx**2+2*dy**2)*(p[3:-1,2:-2]+p[1:-3,2:-2]) \
        -dx**2*dy**2/(2*dx**2+2*dy**2)*(rho*(1/dt*((u[2:-2,3:-1]-u[2:-2,1:-3])/2/dx+(v[3:-1,2:-2]-v[1:-3,2:-2])/2/dy) \
        -u[2:-2,2:-2]*(u[2:-2,1:-3]-2*u[2:-2,2:-2]+u[2:-2,3:-1])/dx**2-((u[2:-2,3:-1]-u[2:-2,1:-3])/2/dx)**2\
          -((v[3:-1,2:-2]-v[1:-3,2:-2])/2/dy)**2-v[2:-2,2:-2]*(v[1:-3,2:-2]-2*v[2:-2,2:-2]+v[3:-1,2:-2])/dy**2 \
          -(u[2:-2,3:-1]*(v[3:-1,3:-1]-v[1:-3,3:-1])/4/dy/dx+v[2:-2,3:-1]*(u[3:-1,3:-1]-u[1:-3,3:-1])/4/dy/dx) \
          +u[2:-2,1:-3]*(v[3:-1,1:-3]-v[1:-3,1:-3])/4/dy/dx+v[2:-2,1:-3]*(u[3:-1,1:-3]-u[1:-3,1:-3])/4/dy/dx \
          -2*(u[3:-1,2:-2]-u[1:-3,2:-2])*(v[2:-2,3:-1]-v[2:-2,1:-3])/4/dy/dx)\
          +mu*((u[3:-1,3:-1]-2*u[2:-2,3:-1]+u[1:-3,3:-1])/2/dx/dy**2-(u[3:-1,1:-3]-2*u[2:-2,1:-3]+u[1:-3,1:-3])/2/dx/dy**2 \
       +(v[3:-1,3:-1]-2*v[3:-1,2:-2]+v[3:-1,1:-3])/2/dy/dx**2-(v[1:-3,3:-1]-2*v[1:-3,2:-2]+v[1:-3,1:-3])/2/dy/dx**2 \
       +(u[2:-2,4:]-2*u[2:-2,3:-1]+2*u[2:-2,1:-3]-u[2:-2,:-4])/2/dx**3\
       +(v[4:,2:-2]-2*v[3:-1,2:-2]+2*v[1:-3,2:-2]-v[:-4,2:-2])/2/dy**3))
        
        # Enforce dp=0 BCs
        if dp_zero[0]==1:
            pn[:,0]=pn[:,1]
        if dp_zero[1]==1:
            pn[:,-1]=pn[:,-2]
        if dp_zero[2]==1:
            pn[0,:]=pn[1,:]
        if dp_zero[3]==1:
            pn[-1,:]=pn[-2,:]
        # Periodic BCs (pairs are implied)
        if dp_zero[0]==2 or dp_zero[1]==2:
            st=1
            en=-3
            sin=0
            pn[st:en,sin]=dy**2/(2*dx**2+2*dy**2)*(p[st:en,sin+1]+p[st:en,sin-1])+dx**2/(2*dx**2+2*dy**2)*(p[st+1:en+1,sin]+p[st-1:en-1,sin]) \
            -dx**2*dy**2/(2*dx**2+2*dy**2)*(rho*(1/dt*((u[st:en,sin+1]-u[st:en,sin-1])/2/dx+(v[st+1:en+1,sin]-v[st-1:en-1,sin])/2/dy) \
            -u[st:en,sin]*(u[st:en,sin-1]-2*u[st:en,sin]+u[st:en,sin+1])/dx**2-((u[st:en,sin+1]-u[st:en,sin-1])/2/dx)**2\
              -((v[st+1:en+1,sin]-v[st-1:en-1,sin])/2/dy)**2-v[st:en,sin]*(v[st-1:en-1,sin]-2*v[st:en,sin]+v[st+1:en+1,sin])/dy**2 \
              -(u[st:en,sin+1]*(v[st+1:en+1,sin+1]-v[st-1:en-1,sin+1])/4/dy/dx+v[st:en,sin+1]*(u[st+1:en+1,sin+1]-u[st-1:en-1,sin+1])/4/dy/dx) \
              +u[st:en,sin-1]*(v[st+1:en+1,sin-1]-v[st-1:en-1,sin-1])/4/dy/dx+v[st:en,sin-1]*(u[st+1:en+1,sin-1]-u[st-1:en-1,sin-1])/4/dy/dx \
              -2*(u[st+1:en+1,sin]-u[st-1:en-1,sin])*(v[st:en,sin+1]-v[st:en,sin-1])/4/dy/dx)\
              +mu*((u[st+1:en+1,sin+1]-2*u[st:en,sin+1]+u[st-1:en-1,sin+1])/2/dx/dy**2-(u[st+1:en+1,sin-1]-2*u[st:en,sin-1]+u[st-1:en-1,sin-1])/2/dx/dy**2 \
               +(v[st+1:en+1,sin+1]-2*v[st+1:en+1,sin]+v[st+1:en+1,sin-1])/2/dy/dx**2-(v[st-1:en-1,sin+1]-2*v[st-1:en-1,sin]+v[st-1:en-1,sin-1])/2/dy/dx**2 \
               +(u[st:en,sin+3]-3*u[st:en,sin+2]+3*u[st:en,sin+1]-u[st:en,sin])/dx**3\
               +(v[st+3:,sin]-3*v[st+2:en+2,sin]+3*v[st+1:en+1,sin]-v[st:en,sin])/dy**3))
            st=-3
            en=-1
            pn[st:en,sin]=dy**2/(2*dx**2+2*dy**2)*(p[st:en,sin+1]+p[st:en,sin-1])+dx**2/(2*dx**2+2*dy**2)*(p[st+1:,sin]+p[st-1:en-1,sin]) \
            -dx**2*dy**2/(2*dx**2+2*dy**2)*(rho*(1/dt*((u[st:en,sin+1]-u[st:en,sin-1])/2/dx+(v[st+1:,sin]-v[st-1:en-1,sin])/2/dy) \
            -u[st:en,sin]*(u[st:en,sin-1]-2*u[st:en,sin]+u[st:en,sin+1])/dx**2-((u[st:en,sin+1]-u[st:en,sin-1])/2/dx)**2\
              -((v[st+1:,sin]-v[st-1:en-1,sin])/2/dy)**2-v[st:en,sin]*(v[st-1:en-1,sin]-2*v[st:en,sin]+v[st+1:,sin])/dy**2 \
              -(u[st:en,sin+1]*(v[st+1:,sin+1]-v[st-1:en-1,sin+1])/4/dy/dx+v[st:en,sin+1]*(u[st+1:,sin+1]-u[st-1:en-1,sin+1])/4/dy/dx) \
              +u[st:en,sin-1]*(v[st+1:,sin-1]-v[st-1:en-1,sin-1])/4/dy/dx+v[st:en,sin-1]*(u[st+1:,sin-1]-u[st-1:en-1,sin-1])/4/dy/dx \
              -2*(u[st+1:,sin]-u[st-1:en-1,sin])*(v[st:en,sin+1]-v[st:en,sin-1])/4/dy/dx)\
              +mu*((u[st+1:,sin+1]-2*u[st:en,sin+1]+u[st-1:en-1,sin+1])/2/dx/dy**2-(u[st+1:,sin-1]-2*u[st:en,sin-1]+u[st-1:en-1,sin-1])/2/dx/dy**2 \
               +(v[st+1:,sin+1]-2*v[st+1:,sin]+v[st+1:,sin-1])/2/dy/dx**2-(v[st-1:en-1,sin+1]-2*v[st-1:en-1,sin]+v[st-1:en-1,sin-1])/2/dy/dx**2 \
               +(u[st:en,sin+3]-3*u[st:en,sin+2]+3*u[st:en,sin+1]-u[st:en,sin])/dx**3\
               -(v[st-3:en-3,sin]-3*v[st-2:en-2,sin]+3*v[st-1:en-1,sin]-v[st:en,sin])/dy**3))
            # Large x
            st=-3
            en=-1
            sin=-1
            pn[st:en,sin]=dy**2/(2*dx**2+2*dy**2)*(p[st:en,sin+1]+p[st:en,sin-1])+dx**2/(2*dx**2+2*dy**2)*(p[st+1:,sin]+p[st-1:en-1,sin]) \
            -dx**2*dy**2/(2*dx**2+2*dy**2)*(rho*(1/dt*((u[st:en,sin+1]-u[st:en,sin-1])/2/dx+(v[st+1:,sin]-v[st-1:en-1,sin])/2/dy) \
            -u[st:en,sin]*(u[st:en,sin-1]-2*u[st:en,sin]+u[st:en,sin+1])/dx**2-((u[st:en,sin+1]-u[st:en,sin-1])/2/dx)**2\
              -((v[st+1:,sin]-v[st-1:en-1,sin])/2/dy)**2-v[st:en,sin]*(v[st-1:en-1,sin]-2*v[st:en,sin]+v[st+1:,sin])/dy**2 \
              -(u[st:en,sin+1]*(v[st+1:,sin+1]-v[st-1:en-1,sin+1])/4/dy/dx+v[st:en,sin+1]*(u[st+1:,sin+1]-u[st-1:en-1,sin+1])/4/dy/dx) \
              +u[st:en,sin-1]*(v[st+1:,sin-1]-v[st-1:en-1,sin-1])/4/dy/dx+v[st:en,sin-1]*(u[st+1:,sin-1]-u[st-1:en-1,sin-1])/4/dy/dx \
              -2*(u[st+1:,sin]-u[st-1:en-1,sin])*(v[st:en,sin+1]-v[st:en,sin-1])/4/dy/dx)\
              +mu*((u[st+1:,sin+1]-2*u[st:en,sin+1]+u[st-1:en-1,sin+1])/2/dx/dy**2-(u[st+1:,sin-1]-2*u[st:en,sin-1]+u[st-1:en-1,sin-1])/2/dx/dy**2 \
               +(v[st+1:,sin+1]-2*v[st+1:,sin]+v[st+1:,sin-1])/2/dy/dx**2-(v[st-1:en-1,sin+1]-2*v[st-1:en-1,sin]+v[st-1:en-1,sin-1])/2/dy/dx**2 \
               -(u[st:en,sin-3]-3*u[st:en,sin-2]+3*u[st:en,sin-1]-u[st:en,sin])/dx**3\
               -(v[st-3:en-3,sin]-3*v[st-2:en-2,sin]+3*v[st-1:en-1,sin]-v[st:en,sin])/dy**3))
            st=1
            en=-3
            pn[st:en,sin]=dy**2/(2*dx**2+2*dy**2)*(p[st:en,sin+1]+p[st:en,sin-1])+dx**2/(2*dx**2+2*dy**2)*(p[st+1:en+1,sin]+p[st-1:en-1,sin]) \
            -dx**2*dy**2/(2*dx**2+2*dy**2)*(rho*(1/dt*((u[st:en,sin+1]-u[st:en,sin-1])/2/dx+(v[st+1:en+1,sin]-v[st-1:en-1,sin])/2/dy) \
            -u[st:en,sin]*(u[st:en,sin-1]-2*u[st:en,sin]+u[st:en,sin+1])/dx**2-((u[st:en,sin+1]-u[st:en,sin-1])/2/dx)**2\
              -((v[st+1:en+1,sin]-v[st-1:en-1,sin])/2/dy)**2-v[st:en,sin]*(v[st-1:en-1,sin]-2*v[st:en,sin]+v[st+1:en+1,sin])/dy**2 \
              -(u[st:en,sin+1]*(v[st+1:en+1,sin+1]-v[st-1:en-1,sin+1])/4/dy/dx+v[st:en,sin+1]*(u[st+1:en+1,sin+1]-u[st-1:en-1,sin+1])/4/dy/dx) \
              +u[st:en,sin-1]*(v[st+1:en+1,sin-1]-v[st-1:en-1,sin-1])/4/dy/dx+v[st:en,sin-1]*(u[st+1:en+1,sin-1]-u[st-1:en-1,sin-1])/4/dy/dx \
              -2*(u[st+1:en+1,sin]-u[st-1:en-1,sin])*(v[st:en,sin+1]-v[st:en,sin-1])/4/dy/dx)\
              +mu*((u[st+1:en+1,sin+1]-2*u[st:en,sin+1]+u[st-1:en-1,sin+1])/2/dx/dy**2-(u[st+1:en+1,sin-1]-2*u[st:en,sin-1]+u[st-1:en-1,sin-1])/2/dx/dy**2 \
               +(v[st+1:en+1,sin+1]-2*v[st+1:en+1,sin]+v[st+1:en+1,sin-1])/2/dy/dx**2-(v[st-1:en-1,sin+1]-2*v[st-1:en-1,sin]+v[st-1:en-1,sin-1])/2/dy/dx**2 \
               -(u[st:en,sin-3]-3*u[st:en,sin-2]+3*u[st:en,sin-1]-u[st:en,sin])/dx**3\
               +(v[st+3:,sin]-3*v[st+2:en+2,sin]+3*v[st+1:en+1,sin]-v[st:en,sin])/dy**3))
     
        if dp_zero[2]==2 or dp_zero[3]==2:
            st=-3
            en=-1
            sin=0
            pn[sin,st:en]=dy**2/(2*dx**2+2*dy**2)*(p[sin,st+1:]+p[sin,st-1:en-1])+dx**2/(2*dx**2+2*dy**2)*(p[sin+1,st:en]+p[sin-1,st:en]) \
            -dx**2*dy**2/(2*dx**2+2*dy**2)*(rho*(1/dt*((u[sin,st+1:]-u[sin,st-1:en-1])/2/dx+(v[sin+1,st:en]-v[sin-1,st:en])/2/dy) \
            -u[sin,st:en]*(u[sin,st-1:en-1]-2*u[sin,st:en]+u[sin,st+1:])/dx**2-((u[sin,st+1:]-u[sin,st-1:en-1])/2/dx)**2\
              -((v[sin+1,st:en]-v[sin-1,st:en])/2/dy)**2-v[sin,st:en]*(v[sin-1,st:en]-2*v[sin,st:en]+v[sin+1,st:en])/dy**2 \
              -(u[sin,st+1:]*(v[sin+1,st+1:]-v[sin-1,st+1:])/4/dy/dx+v[sin,st+1:]*(u[sin+1,st+1:]-u[sin-1,st+1:])/4/dy/dx) \
              +u[sin,st-1:en-1]*(v[sin+1,st-1:en-1]-v[sin-1,st-1:en-1])/4/dy/dx+v[sin,st-1:en-1]*(u[sin+1,st-1:en-1]-u[sin-1,st-1:en-1])/4/dy/dx \
              -2*(u[sin+1,st:en]-u[sin-1,st:en])*(v[sin,st+1:]-v[sin,st-1:en-1])/4/dy/dx)\
              +mu*((u[sin+1,st+1:]-2*u[sin,st+1:]+u[sin-1,st+1:])/2/dx/dy**2-(u[sin+1,st-1:en-1]-2*u[sin,st-1:en-1]+u[sin-1,st-1:en-1])/2/dx/dy**2 \
               +(v[sin+1,st+1:]-2*v[sin+1,st:en]+v[sin+1,st-1:en-1])/2/dy/dx**2-(v[sin-1,st+1:]-2*v[sin-1,st:en]+v[sin-1,st-1:en-1])/2/dy/dx**2 \
               -(u[sin,st-3:en-3]-3*u[sin,st-2:en-2]+3*u[sin,st-1:en-1]-u[sin,st:en])/dx**3\
               +(v[sin+3,st:en]-3*v[sin+2,st:en]+3*v[sin+1,st:en]-v[sin,st:en])/dy**3))
            st=1
            en=-3
            pn[sin,st:en]=dy**2/(2*dx**2+2*dy**2)*(p[sin,st+1:en+1]+p[sin,st-1:en-1])+dx**2/(2*dx**2+2*dy**2)*(p[sin+1,st:en]+p[sin-1,st:en]) \
            -dx**2*dy**2/(2*dx**2+2*dy**2)*(rho*(1/dt*((u[sin,st+1:en+1]-u[sin,st-1:en-1])/2/dx+(v[sin+1,st:en]-v[sin-1,st:en])/2/dy) \
            -u[sin,st:en]*(u[sin,st-1:en-1]-2*u[sin,st:en]+u[sin,st+1:en+1])/dx**2-((u[sin,st+1:en+1]-u[sin,st-1:en-1])/2/dx)**2\
              -((v[sin+1,st:en]-v[sin-1,st:en])/2/dy)**2-v[sin,st:en]*(v[sin-1,st:en]-2*v[sin,st:en]+v[sin+1,st:en])/dy**2 \
              -(u[sin,st+1:en+1]*(v[sin+1,st+1:en+1]-v[sin-1,st+1:en+1])/4/dy/dx+v[sin,st+1:en+1]*(u[sin+1,st+1:en+1]-u[sin-1,st+1:en+1])/4/dy/dx) \
              +u[sin,st-1:en-1]*(v[sin+1,st-1:en-1]-v[sin-1,st-1:en-1])/4/dy/dx+v[sin,st-1:en-1]*(u[sin+1,st-1:en-1]-u[sin-1,st-1:en-1])/4/dy/dx \
              -2*(u[sin+1,st:en]-u[sin-1,st:en])*(v[sin,st+1:en+1]-v[sin,st-1:en-1])/4/dy/dx)\
              +mu*((u[sin+1,st+1:en+1]-2*u[sin,st+1:en+1]+u[sin-1,st+1:en+1])/2/dx/dy**2-(u[sin+1,st-1:en-1]-2*u[sin,st-1:en-1]+u[sin-1,st-1:en-1])/2/dx/dy**2 \
               +(v[sin+1,st+1:en+1]-2*v[sin+1,st:en]+v[sin+1,st-1:en-1])/2/dy/dx**2-(v[sin-1,st+1:en+1]-2*v[sin-1,st:en]+v[sin-1,st-1:en-1])/2/dy/dx**2 \
               +(u[sin,st+3:]-3*u[sin,st+2:en+2]+3*u[sin,st+1:en+1]-u[sin,st:en])/dx**3\
               +(v[sin+3,st:en]-3*v[sin+2,st:en]+3*v[sin+1,st:en]-v[sin,st:en])/dy**3))
            # Large y
            st=-3
            en=-1
            sin=-1
            pn[sin,st:en]=dy**2/(2*dx**2+2*dy**2)*(p[sin,st+1:]+p[sin,st-1:en-1])+dx**2/(2*dx**2+2*dy**2)*(p[sin+1,st:en]+p[sin-1,st:en]) \
            -dx**2*dy**2/(2*dx**2+2*dy**2)*(rho*(1/dt*((u[sin,st+1:]-u[sin,st-1:en-1])/2/dx+(v[sin+1,st:en]-v[sin-1,st:en])/2/dy) \
            -u[sin,st:en]*(u[sin,st-1:en-1]-2*u[sin,st:en]+u[sin,st+1:])/dx**2-((u[sin,st+1:]-u[sin,st-1:en-1])/2/dx)**2\
              -((v[sin+1,st:en]-v[sin-1,st:en])/2/dy)**2-v[sin,st:en]*(v[sin-1,st:en]-2*v[sin,st:en]+v[sin+1,st:en])/dy**2 \
              -(u[sin,st+1:]*(v[sin+1,st+1:]-v[sin-1,st+1:])/4/dy/dx+v[sin,st+1:]*(u[sin+1,st+1:]-u[sin-1,st+1:])/4/dy/dx) \
              +u[sin,st-1:en-1]*(v[sin+1,st-1:en-1]-v[sin-1,st-1:en-1])/4/dy/dx+v[sin,st-1:en-1]*(u[sin+1,st-1:en-1]-u[sin-1,st-1:en-1])/4/dy/dx \
              -2*(u[sin+1,st:en]-u[sin-1,st:en])*(v[sin,st+1:]-v[sin,st-1:en-1])/4/dy/dx)\
              +mu*((u[sin+1,st+1:]-2*u[sin,st+1:]+u[sin-1,st+1:])/2/dx/dy**2-(u[sin+1,st-1:en-1]-2*u[sin,st-1:en-1]+u[sin-1,st-1:en-1])/2/dx/dy**2 \
               +(v[sin+1,st+1:]-2*v[sin+1,st:en]+v[sin+1,st-1:en-1])/2/dy/dx**2-(v[sin-1,st+1:]-2*v[sin-1,st:en]+v[sin-1,st-1:en-1])/2/dy/dx**2 \
               -(u[sin,st-3:en-3]-3*u[sin,st-2:en-2]+3*u[sin,st-1:en-1]-u[sin,st:en])/dx**3\
               -(v[sin-3,st:en]-3*v[sin-2,st:en]+3*v[sin-1,st:en]-v[sin,st:en])/dy**3))
            st=1
            en=-3
            pn[sin,st:en]=dy**2/(2*dx**2+2*dy**2)*(p[sin,st+1:]+p[sin,st-1:en-1])+dx**2/(2*dx**2+2*dy**2)*(p[sin+1,st:en]+p[sin-1,st:en]) \
            -dx**2*dy**2/(2*dx**2+2*dy**2)*(rho*(1/dt*((u[sin,st+1:]-u[sin,st-1:en-1])/2/dx+(v[sin+1,st:en]-v[sin-1,st:en])/2/dy) \
            -u[sin,st:en]*(u[sin,st-1:en-1]-2*u[sin,st:en]+u[sin,st+1:])/dx**2-((u[sin,st+1:]-u[sin,st-1:en-1])/2/dx)**2\
              -((v[sin+1,st:en]-v[sin-1,st:en])/2/dy)**2-v[sin,st:en]*(v[sin-1,st:en]-2*v[sin,st:en]+v[sin+1,st:en])/dy**2 \
              -(u[sin,st+1:]*(v[sin+1,st+1:]-v[sin-1,st+1:])/4/dy/dx+v[sin,st+1:]*(u[sin+1,st+1:]-u[sin-1,st+1:])/4/dy/dx) \
              +u[sin,st-1:en-1]*(v[sin+1,st-1:en-1]-v[sin-1,st-1:en-1])/4/dy/dx+v[sin,st-1:en-1]*(u[sin+1,st-1:en-1]-u[sin-1,st-1:en-1])/4/dy/dx \
              -2*(u[sin+1,st:en]-u[sin-1,st:en])*(v[sin,st+1:]-v[sin,st-1:en-1])/4/dy/dx)\
              +mu*((u[sin+1,st+1:]-2*u[sin,st+1:]+u[sin-1,st+1:])/2/dx/dy**2-(u[sin+1,st-1:en-1]-2*u[sin,st-1:en-1]+u[sin-1,st-1:en-1])/2/dx/dy**2 \
               +(v[sin+1,st+1:]-2*v[sin+1,st:en]+v[sin+1,st-1:en-1])/2/dy/dx**2-(v[sin-1,st+1:]-2*v[sin-1,st:en]+v[sin-1,st-1:en-1])/2/dy/dx**2 \
               +(u[sin,st+3:]-3*u[sin,st+2:en+2]+3*u[sin,st+1:en+1]-u[sin,st:en])/dx**3\
               -(v[sin-3,st:en]-3*v[sin-2,st:en]+3*v[sin-1,st:en]-v[sin,st:en])/dy**3))

          # Convergence check
        diff=numpy.sum(numpy.abs(p[:]-pn[:]))/numpy.sum(numpy.abs(p[:]))
        if YesRes:
            print(diff)
        p=pn.copy()
        count=count+1
    if count==1000:
        print 'Convergence problems resolving the pressure distribution'
        print 'Residuals: %.6f'%diff
        error=1
    # Pressure clipping
#    if numpy.amax(p[:])<10**(-5):
#        p[:,:]=0
    
    return p,error

def SolveVel(Nt,u,v,p,sources,dxyt,mat_prop,conv,BC_vel_press,print_out):
    rho,mu=mat_prop
    dx,dy,dt=dxyt
    BC_vel,BC_press=BC_vel_press
    YesPrint,YesRes=print_out
    dpx,dpy,gx,gy=sources
    count=1
    diff_SSu=10
    diff_SSv=10
    error2=0
    un=numpy.empty_like(u)
    vn=numpy.empty_like(v)
#    while (diff_SS>conv) and (count<1000): # Use for steady state solving    
    for i in range(Nt):
    
#        if 1.0*i/100==1:
#            dt=dt*10
#            print'Time step changed to %.2f'%dt
        
        if YesPrint:
            print 'Time step %i \n'%i
        if YesRes:
            print 'Pressure residuals:'
        
        p,error=ResolvePress(p, u, v, (dx, dy, dt), (rho, mu), conv, YesRes, pres_BCs)
        if error==1:
            print 'Run aborted at time step %i'%i
            error2=1
            break
        # Solve momentum equations (explicit, first order for time)
        un=u.copy()
        vn=v.copy()
        u[1:-1,1:-1]=dt/(2*rho*dx)*(p[1:-1,:-2]-p[1:-1,2:]) \
          +un[1:-1,2:]*(dt*nu/dx**2-dt/2/dx*un[1:-1,1:-1]) \
          +un[1:-1,:-2]*(dt*nu/dx**2+dt/2/dx*un[1:-1,1:-1]) \
          +un[2:,1:-1]*(dt*nu/dy**2-dt/2/dy*vn[1:-1,1:-1]) \
          +un[:-2,1:-1]*(dt*nu/dy**2+dt/2/dy*vn[1:-1,1:-1]) \
          +un[1:-1,1:-1]*(1-2*nu*dt*(1/dx**2+1/dy**2)) + dt*(gx-dpx/rho)
        
        v[1:-1,1:-1]=dt/(2*rho*dy)*(p[:-2,1:-1]-p[2:,1:-1]) \
          +vn[1:-1,2:]*(dt*nu/dx**2-dt/2/dx*un[1:-1,1:-1]) \
          +vn[1:-1,:-2]*(dt*nu/dx**2+dt/2/dx*vn[1:-1,1:-1]) \
          +vn[2:,1:-1]*(dt*nu/dy**2-dt/2/dy*vn[1:-1,1:-1]) \
          +vn[:-2,1:-1]*(dt*nu/dy**2+dt/2/dy*vn[1:-1,1:-1]) \
          +vn[1:-1,1:-1]*(1-2*dt*nu*(1/dx**2+1/dy**2)) + dt*(gy-dpy/rho)
        
        # Periodic BCs (across x)
        if (BC_vel[0]==2) or (BC_vel[1]==2):
            u[1:-1,-1]=dt/(2*rho*dx)*(p[1:-1,-2]-p[1:-1,0]) \
              +un[1:-1,0]*(dt*nu/dx**2-dt/2/dx*un[1:-1,-1]) \
              +un[1:-1,-2]*(dt*nu/dx**2+dt/2/dx*un[1:-1,-1]) \
              +un[2:,-1]*(dt*nu/dy**2-dt/2/dy*vn[1:-1,-1]) \
              +un[:-2,-1]*(dt*nu/dy**2+dt/2/dy*vn[1:-1,-1]) \
              +un[1:-1,-1]*(1-2*nu*dt*(1/dx**2+1/dy**2)) + dt*(gx-dpx/rho)
            
            u[1:-1,0]=dt/(2*rho*dx)*(p[1:-1,-1]-p[1:-1,1]) \
              +un[1:-1,1]*(dt*nu/dx**2-dt/2/dx*un[1:-1,0]) \
              +un[1:-1,-1]*(dt*nu/dx**2+dt/2/dx*un[1:-1,0]) \
              +un[2:,0]*(dt*nu/dy**2-dt/2/dy*vn[1:-1,0]) \
              +un[:-2,0]*(dt*nu/dy**2+dt/2/dy*vn[1:-1,0]) \
              +un[1:-1,0]*(1-2*nu*dt*(1/dx**2+1/dy**2)) + dt*(gx-dpx/rho)
            
            v[1:-1,-1]=dt/(2*rho*dy)*(p[:-2,-1]-p[2:,-1]) \
              +vn[1:-1,0]*(dt*nu/dx**2-dt/2/dx*un[1:-1,-1]) \
              +vn[1:-1,-2]*(dt*nu/dx**2+dt/2/dx*vn[1:-1,-1]) \
              +vn[2:,-1]*(dt*nu/dy**2-dt/2/dy*vn[1:-1,-1]) \
              +vn[:-2,-1]*(dt*nu/dy**2+dt/2/dy*vn[1:-1,-1]) \
              +vn[1:-1,-1]*(1-2*dt*nu*(1/dx**2+1/dy**2)) + dt*(gy-dpy/rho)
            
            v[1:-1,0]=dt/(2*rho*dy)*(p[:-2,0]-p[2:,0]) \
              +vn[1:-1,1]*(dt*nu/dx**2-dt/2/dx*un[1:-1,0]) \
              +vn[1:-1,-1]*(dt*nu/dx**2+dt/2/dx*vn[1:-1,0]) \
              +vn[2:,0]*(dt*nu/dy**2-dt/2/dy*vn[1:-1,0]) \
              +vn[:-2,0]*(dt*nu/dy**2+dt/2/dy*vn[1:-1,0]) \
              +vn[1:-1,0]*(1-2*dt*nu*(1/dx**2+1/dy**2)) + dt*(gy-dpy/rho)
        
        # Periodic BCs (across y)
        if (BC_vel[2]==2) or (BC_vel[3]==2):
            u[-1,1:-1]=dt/(2*rho*dx)*(p[-1,:-2]-p[-1,2:]) \
              +un[-1,2:]*(dt*nu/dx**2-dt/2/dx*un[-1,1:-1]) \
              +un[-1,:-2]*(dt*nu/dx**2+dt/2/dx*un[-1,1:-1]) \
              +un[0,1:-1]*(dt*nu/dy**2-dt/2/dy*vn[-1,1:-1]) \
              +un[-2,1:-1]*(dt*nu/dy**2+dt/2/dy*vn[-1,1:-1]) \
              +un[-1,1:-1]*(1-2*nu*dt*(1/dx**2+1/dy**2)) + dt*(gx-dpx/rho)
            u[0,1:-1]=dt/(2*rho*dx)*(p[0,:-2]-p[0,2:]) \
              +un[0,2:]*(dt*nu/dx**2-dt/2/dx*un[0,1:-1]) \
              +un[0,:-2]*(dt*nu/dx**2+dt/2/dx*un[0,1:-1]) \
              +un[1,1:-1]*(dt*nu/dy**2-dt/2/dy*vn[0,1:-1]) \
              +un[-1,1:-1]*(dt*nu/dy**2+dt/2/dy*vn[0,1:-1]) \
              +un[0,1:-1]*(1-2*nu*dt*(1/dx**2+1/dy**2)) + dt*(gx-dpx/rho)
            
            v[-1,1:-1]=dt/(2*rho*dy)*(p[-2,1:-1]-p[0,1:-1]) \
              +vn[-1,2:]*(dt*nu/dx**2-dt/2/dx*un[-1,1:-1]) \
              +vn[-1,:-2]*(dt*nu/dx**2+dt/2/dx*vn[-1,1:-1]) \
              +vn[0,1:-1]*(dt*nu/dy**2-dt/2/dy*vn[-1,1:-1]) \
              +vn[-2,1:-1]*(dt*nu/dy**2+dt/2/dy*vn[-1,1:-1]) \
              +vn[-1,1:-1]*(1-2*dt*nu*(1/dx**2+1/dy**2)) + dt*(gy-dpy/rho)
            v[0,1:-1]=dt/(2*rho*dy)*(p[-1,1:-1]-p[1,1:-1]) \
              +vn[0,2:]*(dt*nu/dx**2-dt/2/dx*un[0,1:-1]) \
              +vn[0,:-2]*(dt*nu/dx**2+dt/2/dx*vn[0,1:-1]) \
              +vn[1,1:-1]*(dt*nu/dy**2-dt/2/dy*vn[0,1:-1]) \
              +vn[-1,1:-1]*(dt*nu/dy**2+dt/2/dy*vn[0,1:-1]) \
              +vn[0,1:-1]*(1-2*dt*nu*(1/dx**2+1/dy**2)) + dt*(gy-dpy/rho)
        
        diff_SSu=numpy.sum(numpy.abs(u[:]-un[:]))/numpy.sum(numpy.abs(u[:]))
        diff_SSv=numpy.sum(numpy.abs(v[:]-vn[:]))/numpy.sum(numpy.abs(v[:]))
        count+=1
        
    return u,v,p,error2


#-------------------------------- Setup
L=2.0 # Length (x coordinate max)
W=2.0 # Width
gx=0 # Gravity component in x (if neglected, use 0)
gy=0 # Gravity component in y (if neglected, use 0)
dpx=0 # Pressure gradient in x
dpy=0 # Pressure gradient in y
Nx=41 # Number of nodes in x
Ny=41 # Number of nodes in y
rho=1.0 # Density of fluid (kg/m^3) Water 998, air 1.2
mu=0.1*rho # Dynamic viscosity of fluid (Pa s) Water 10**(-3), air 1.8*10**(-5)
dt=0.001 # Time step size (s)
Nt=10 # Number of time steps

u=numpy.zeros((Ny, Nx))
v=numpy.zeros((Ny, Nx))
p=numpy.zeros((Ny, Nx))
x=numpy.linspace(0, L, Nx)
y=numpy.linspace(0, W, Ny)
X,Y=numpy.meshgrid(x,y)
dx=L/(Nx-1)
dy=W/(Ny-1)
nu=mu/rho
dxyt=(dx,dy,dt)
mat_prop=(rho,mu)
src=(dpx,dpy,gx,gy)

# Convergence
conv=0.01 # convergence criteria for pressure and velocity
YesPrint=1  # Bool-print time step data
YesRes=0 # Bool-print residuals data at each time step

# Boundary conditions
u[:,0]=0
u[:,-1]=0
#u[0,:]=0
u[-1,:]=1
v[:,-1]=0
v[:,0]=0
#v[0,:]=0
#v[-1,:]=0

#p[-1,:]=0
#p[-4,:]=1
vel_BCs=(0,0,0,0) # BC types for velocities
pres_BCs=(1,1,1,0) # BC types for pressure

BCs=(vel_BCs,pres_BCs)
msgs=(YesPrint,YesRes)

#-------------------------------- Solve

u,v,p,error=SolveVel(Nt,u,v,p,src,dxyt,mat_prop,conv,BCs,msgs)


pl=2 # Plot every "pl" number of points on quiver plot
fig = pyplot.figure(figsize=(11,7), dpi=100)
# plotting the pressure field as a contour
pyplot.contourf(X, Y, p, alpha=0.5, cmap=cm.viridis)  
pyplot.colorbar()
# plotting the pressure field outlines
pyplot.contour(X, Y, p, cmap=cm.viridis)  
# plotting velocity field
pyplot.quiver(X[::pl, ::pl], Y[::pl, ::pl], u[::pl, ::pl], v[::pl, ::pl]) 
pyplot.xlabel('X')
pyplot.ylabel('Y');