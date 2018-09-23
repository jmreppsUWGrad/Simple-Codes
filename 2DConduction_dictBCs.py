# -*- coding: utf-8 -*-
"""
Started on Thu Sep 06 20:09:37 2018

@author: Joseph

This script contains solvers for 2D planar conduction:

Features:
    Steady, transient (explicit or implicit)
    Temperature, flux or convective BCs (4 sides customizable)
    Built-in Fourrier number adjustment for stability
    dx and dy don't have to be equal
    CANNOT handle 2 adjacent flux/convective BCs (corners)

THIS REVISION:
    Customizable BC along length of boundary
    Fourrier number adjustment routine needs re-coding

NOTE: Temperature array setup so rows are y and columns 
are x, but row indices start at the lowest y coordinate
and increase

Variables:
    T: Temperature array
    Nx, Ny: Number of nodes in x and y directions respectively
    dx, dy, dt: Discretization size for x, y and time respectively
    rho, Cp, k: Material properties-density, specific heat, thermal conductivity
    Fo: Fourrier number defined Fo=k*dt/(rho*Cp*dx*dy)
    Bi: Biot number defined either Bix=hx*dx/k OR Biy=hy*dy/k
    alpha: Relaxation parameter

Function inputs:
    dxy: array containing dx and dy
    conv_param: array containing convergence criteria and relaxation parameter
    BC_setup: 4 number array to designate BCs at x1, x2, y1, y2 boundaries respectively
            1-temp, 2-flux, 3-convective 
            e.g. 1212-temp BCs at smallest x and y, flux at largest x and y
    solver_type: 0-Steady, 1-explicit trans, 2-implicit trans
    bc1- array containing BC info for smallest x
    bc2- array containing BC info for largest x
    bc3- array containing BC info for smallest y
    bc4- array containing BC info for largest y

"""

# Libraries
import numpy
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D

# Steady solver function incl BCs
def SteadySolve(T, dxy, k, conv_param, BC_info):
    dx,dy=dxy
    conv,alpha=conv_param
    error=0
    diff=10
    count=1
    # Assign temperature BCs if applicable
    for i in range(len(BC_info['BCx1'])/3):
        if BC_info['BCx1'][3*i]=='T':
            st=BC_info['BCx1'][2+3*i][0]
            en=BC_info['BCx1'][2+3*i][1]
            T[st:en,0]=BC_info['BCx1'][1+3*i]
            if len(BC_info['BCx1'])/3-i==1:
                T[-1,0]=BC_info['BCx1'][-2]
    for i in range(len(BC_info['BCx2'])/3):
        if BC_info['BCx2'][3*i]=='T':
            st=BC_info['BCx2'][2+3*i][0]
            en=BC_info['BCx2'][2+3*i][1]
            T[st:en,-1]=BC_info['BCx2'][1+3*i]
            if len(BC_info['BCx2'])/3-i==1:
                T[-1,-1]=BC_info['BCx2'][-2]
    for i in range(len(BC_info['BCy1'])/3):
        if BC_info['BCy1'][3*i]=='T':
            st=BC_info['BCy1'][2+3*i][0]
            en=BC_info['BCy1'][2+3*i][1]
            T[0,st:en]=BC_info['BCy1'][1+3*i]
            if len(BC_info['BCy1'])/3-i==1:
                T[0,-1]=BC_info['BCy1'][-2]
    for i in range(len(BC_info['BCy2'])/3):
        if BC_info['BCy2'][3*i]=='T':
            st=BC_info['BCy2'][2+3*i][0]
            en=BC_info['BCy2'][2+3*i][1]
            T[-1,st:en]=BC_info['BCy2'][1+3*i]
            if len(BC_info['BCy2'])/3-i==1:
                T[-1,-1]=BC_info['BCy2'][-2]
        
    Tc2=T.copy()
    print 'Steady state solver'
    print 'Residuals:'
    while (diff>conv) and (count<1000):
        Tc=T.copy()
        Tc2[1:-1, 1:-1]=(dx**2*(Tc[:-2,1:-1]+Tc[2:,1:-1]) \
        +dy**2*(Tc[1:-1,:-2]+Tc[1:-1,2:])) \
        /(2*dx**2+2*dy**2)
        T[1:-1, 1:-1]=alpha*Tc2[1:-1, 1:-1]+(1-alpha)*Tc[1:-1, 1:-1]
        
        # Apply flux BC if applicable
        for i in range(len(BC_info['BCx1'])/3):
            if BC_info['BCx1'][0]=='F':
                st=BC_info['BCx1'][2+3*i][0]
                en=BC_info['BCx1'][2+3*i][1]
                q=BC_info['BCx1'][1+3*i]
                T[st:en,0]=(2*q*dy**2*dx/k+2*dy**2*T[st:en,1]\
                         +dx**2*(T[st-1:en-1,0]+T[st+1:en+1,0]))\
                         /(2*dy**2+2*dx**2)
                if len(BC_info['BCx1'])/3-i==1:
                    T[-2,0]=(2*q*dy**2*dx/k+2*dy**2*T[-2,1]\
                         +dx**2*(T[-3,0]+T[-1,0]))\
                         /(2*dy**2+2*dx**2)
        for i in range(len(BC_info['BCx2'])/3):
            if BC_info['BCx2'][0]=='F':
                st=BC_info['BCx2'][2+3*i][0]
                en=BC_info['BCx2'][2+3*i][1]
                q=BC_info['BCx2'][1+3*i]
                T[st:en,-1]=(2*q*dy**2*dx/k+2*dy**2*T[st:en,-2]\
                         +dx**2*(T[st-1:en-1,-1]+T[st+1:en+1,-1]))\
                         /(2*dy**2+2*dx**2)
                if len(BC_info['BCx2'])/3-i==1:
                    T[-2,-1]=(2*q*dy**2*dx/k+2*dy**2*T[-2,-2]\
                         +dx**2*(T[-3,-1]+T[-1,-1]))\
                         /(2*dy**2+2*dx**2)
        for i in range(len(BC_info['BCy1'])/3):
            if BC_info['BCy1'][0]=='F':
                st=BC_info['BCy1'][2+3*i][0]
                en=BC_info['BCy1'][2+3*i][1]
                q=BC_info['BCy1'][1+3*i]
                T[0,st:en]=(2*q*dx**2*dy/k+2*dx**2*T[1,st:en]\
                         +dy**2*(T[0,st-1:en-1]+T[0,st+1:en+1]))\
                         /(2*dx**2+2*dy**2)
                if len(BC_info['BCy1'])/3-i==1:
                    T[0,-2]=(2*q*dx**2*dy/k+2*dx**2*T[1,-2]\
                         +dy**2*(T[0,-3]+T[0,-1]))\
                         /(2*dx**2+2*dy**2)
        for i in range(len(BC_info['BCy2'])/3):
            if BC_info['BCy2'][0]=='F':
                st=BC_info['BCy1'][2+3*i][0]
                en=BC_info['BCy1'][2+3*i][1]
                q=BC_info['BCy2'][1+3*i]
                T[-1,st:en]=(2*q*dx**2*dy/k+2*dx**2*T[-2,st:en]\
                         +dy**2*(T[-1,st-1:en-1]+T[-1,st+1:en+1]))\
                         /(2*dx**2+2*dy**2)
                if len(BC_info['BCy2'])/3-i==1:
                    T[-1,-2]=(2*q*dx**2*dy/k+2*dx**2*T[-2,-2]\
                         +dy**2*(T[-1,-3]+T[-1,-1]))\
                         /(2*dx**2+2*dy**2)
        # Apply convective BC if applicable
        for i in range(len(BC_info['BCx1'])/3):
            if BC_info['BCx1'][0]=='C':
                st=BC_info['BCx1'][2+3*i][0]
                en=BC_info['BCx1'][2+3*i][1]
                Bi=BC_info['BCx1'][1+3*i][0]*dx/k
                T[st:en,0]=(2*Bi*dy**2*BC_info['BCx1'][1+3*i][1]+2*dy**2*T[st:en,1]\
                 +dx**2*(T[st-1:en-1,0]+T[st+1:en+1,0]))/(2*dy**2+2*dx**2+2*Bi*dy**2)
                if len(BC_info['BCx1'])/3-i==1:
                    T[-2,0]=(2*Bi*dy**2*BC_info['BCx1'][1+3*i][1]+2*dy**2*T[-2,1]\
                     +dx**2*(T[-3,0]+T[-1,0]))/(2*dy**2+2*dx**2+2*Bi*dy**2)
        for i in range(len(BC_info['BCx2'])/3):
            if BC_info['BCx2'][0]=='C':
                st=BC_info['BCx2'][2+3*i][0]
                en=BC_info['BCx2'][2+3*i][1]
                Bi=BC_info['BCx2'][1+3*i][0]*dx/k
                T[st:en,-1]=(2*Bi*dy**2*BC_info['BCx2'][1+3*i][1]+2*dy**2*T[st:en,-2]\
                 +dx**2*(T[st-1:en-1,-1]+T[st+1:en+1,-1]))/(2*dy**2+2*dx**2+2*Bi*dy**2)
                if len(BC_info['BCx2'])/3-i==1:
                    T[-2,-1]=(2*Bi*dy**2*BC_info['BCx2'][1+3*i][1]+2*dy**2*T[-2,-2]\
                     +dx**2*(T[-3,-1]+T[-1,-1]))/(2*dy**2+2*dx**2+2*Bi*dy**2)
        for i in range(len(BC_info['BCy1'])/3):
            if BC_info['BCy1'][0]=='C':
                st=BC_info['BCy1'][2+3*i][0]
                en=BC_info['BCy1'][2+3*i][1]
                Bi=BC_info['BCy1'][1+3*i][0]*dy/k
                T[0,st:en]=(2*Bi*dx**2*BC_info['BCy1'][1+3*i][1]+2*dx**2*T[1,st:en]\
                +dy**2*(T[0,st-1:en-1]+T[0,st+1:en+1]))/(2*dx**2+2*dy**2+2*Bi*dx**2)
                if len(BC_info['BCy1'])/3-i==1:
                    T[0,-2]=(2*Bi*dx**2*BC_info['BCy1'][1+3*i][1]+2*dx**2*T[1,-2]\
                     +dy**2*(T[0,-3]+T[0,-1]))/(2*dx**2+2*dy**2+2*Bi*dx**2)
        for i in range(len(BC_info['BCy2'])/3):
            if BC_info['BCy2'][0]=='C':
                st=BC_info['BCy2'][2+3*i][0]
                en=BC_info['BCy2'][2+3*i][1]
                Bi=BC_info['BCy2'][1+3*i][0]*dy/k
                T[-1,st:en]=(2*Bi*dx**2*BC_info['BCy2'][1+3*i][1]+2*dx**2*T[-2,st:en]\
                +dy**2*(T[-1,st-1:en-1]+T[-1,st+1:en+1]))/(2*dx**2+2*dy**2+2*Bi*dx**2)
                if len(BC_info['BCy2'])/3-i==1:
                    T[-1,-2]=(2*Bi*dx**2*BC_info['BCy2'][1+3*i][1]+2*dx**2*T[-2,-2]\
                     +dy**2*(T[-1,-3]+T[-1,-1]))/(2*dx**2+2*dy**2+2*Bi*dx**2)
        diff=numpy.sum(numpy.abs(T[:]-Tc[:]))/numpy.sum(numpy.abs(T[:]))
        count=count+1
        print(diff)
    
    if count==1000:
        error=1
    return T, error
    
# Transient solvers
def TransSolve(is_explicit, T, dxy, k, Fo, conv_param, BC_info):
    dx,dy=dxy
    conv,alpha=conv_param
    error=0
    diff=10
    count=1
    Fo_old=Fo
    # Assign temperature BCs if applicable
    for i in range(len(BC_info['BCx1'])/3):
        if BC_info['BCx1'][3*i]=='T':
            st=BC_info['BCx1'][2+3*i][0]
            en=BC_info['BCx1'][2+3*i][1]
            T[st:en,0]=BC_info['BCx1'][1+3*i]
            if len(BC_info['BCx1'])/3-i==1:
                T[-1,0]=BC_info['BCx1'][-2]
    for i in range(len(BC_info['BCx2'])/3):
        if BC_info['BCx2'][3*i]=='T':
            st=BC_info['BCx2'][2+3*i][0]
            en=BC_info['BCx2'][2+3*i][1]
            T[st:en,-1]=BC_info['BCx2'][1+3*i]
            if len(BC_info['BCx2'])/3-i==1:
                T[-1,-1]=BC_info['BCx2'][-2]
    for i in range(len(BC_info['BCy1'])/3):
        if BC_info['BCy1'][3*i]=='T':
            st=BC_info['BCy1'][2+3*i][0]
            en=BC_info['BCy1'][2+3*i][1]
            T[0,st:en]=BC_info['BCy1'][1+3*i]
            if len(BC_info['BCy1'])/3-i==1:
                T[0,-1]=BC_info['BCy1'][-2]
    for i in range(len(BC_info['BCy2'])/3):
        if BC_info['BCy2'][3*i]=='T':
            st=BC_info['BCy2'][2+3*i][0]
            en=BC_info['BCy2'][2+3*i][1]
            T[-1,st:en]=BC_info['BCy2'][1+3*i]
            if len(BC_info['BCy2'])/3-i==1:
                T[-1,-1]=BC_info['BCy2'][-2]
    
    Tc2=T.copy()
    # Explicit solver and flux/convective BCs
    if is_explicit:
        # Apply stability criteria to Fo (temp BCs)
#        Fo=min(Fo, dx*dy/(2*(dy**2+dx**2)))
#        # Apply stability criteria to Fo (convective BCs)
#        if (BC_setup[0]==3) and (BC_setup[1]==3):
#            Bi1=BC_info['BCx1'][1+3*i][0]*dx/k
#            Bi2=BC_info['BCx2'][1+3*i][0]*dx/k
#            Fo=min(Fo, 0.5*dx/((dy**2+2*dx**2)/dy+2*Bi1*dy),\
#                   0.5*dx/((dy**2+2*dx**2)/dy+2*Bi2*dy))
#        elif (BC_setup[0]==3) or (BC_setup[1]==3):
#            if BC_setup[0]==3:
#                Bi=bc1[0]*dx/k
#            else:
#                Bi=bc2[0]*dx/k
#            Fo=min(Fo, 0.5*dx/((dy**2+2*dx**2)/dy+2*Bi*dy))
#        if (BC_setup[2]==3) and (BC_setup[3]==3):
#            Bi1=bc3[0]*dy/k
#            Bi2=bc4[0]*dy/k
#            Fo=min(Fo, 0.5*dy/((dx**2+2*dy**2)/dx+2*Bi1*dx),\
#                   0.5*dy/((dx**2+2*dy**2)/dy+2*Bi2*dx))
#        elif (BC_setup[2]==3) or (BC_setup[3]==3):
#            if BC_setup[2]==3:
#                Bi=bc3[0]*dy/k
#            else:
#                Bi=bc4[0]*dy/k
#            Fo=min(Fo, 0.5*dy/((dx**2+2*dy**2)/dy+2*Bi2*dx))
#        if Fo!=Fo_old:
#            print 'Fourrier number adjusted to %.2f for stability'%Fo
        # Proceed to solve
        Tc=T.copy()
        T[1:-1,1:-1]=(Fo*dy**2*(Tc[1:-1,:-2]+Tc[1:-1,2:])\
            +Fo*dx**2*(Tc[:-2,1:-1]+Tc[2:,1:-1])\
            +(dx*dy-2*Fo*(dx**2+dy**2))*Tc[1:-1,1:-1])/(dx*dy)
        
        #T[1:-1, 1:-1]=alpha*Tc2[1:-1, 1:-1]+(1-alpha)*Tc[1:-1, 1:-1]
        
        # Apply BC for lowest x
        for i in range(len(BC_info['BCx1'])/3):
            # Flux
            if BC_info['BCx1'][3*i]=='F':
                st=BC_info['BCx1'][2+3*i][0]
                en=BC_info['BCx1'][2+3*i][1]
                q=BC_info['BCx1'][1+3*i]
                T[st:en,0]=(2*Fo*q*dy**2*dx/k+2*Fo*dy**2*Tc[st:en,1]\
                 +Fo*dx**2*(Tc[st-1:en-1,0]+Tc[st+1:en+1,0])+(dx*dy\
                 -2*Fo*(dy**2+dx**2))*Tc[st:en,0])/(dx*dy)
                if len(BC_info['BCx1'])/3-i==1:
                    T[-2,0]=(2*Fo*q*dy**2*dx/k+2*Fo*dy**2*Tc[-2,1]\
                     +Fo*dx**2*(Tc[-3,0]+Tc[-1,0])+(dx*dy\
                     -2*Fo*(dy**2+dx**2))*Tc[-2,0])/(dx*dy)
            # Convective
            if BC_info['BCx1'][3*i]=='C':
                st=BC_info['BCx1'][2+3*i][0]
                en=BC_info['BCx1'][2+3*i][1]
                Bi=BC_info['BCx1'][1+3*i][0]*dx/k
                T[st:en,0]=(2*Fo*Bi*dy**2*BC_info['BCx1'][1+3*i][1]+2*Fo*dy**2*Tc[st:en,1]\
                 +2*Fo*dx**2*(Tc[st-1:en-1,0]+Tc[st+1:en+1,0])+(dx*dy\
                 -2*Fo*(dy**2+2*dx**2)-2*Fo*Bi*dy**2)*Tc[st:en,0])/(dx*dy)
                if len(BC_info['BCx1'])/3-i==1:
                    T[-2,0]=(2*Fo*Bi*dy**2*BC_info['BCx1'][1+3*i][1]+2*Fo*dy**2*Tc[-2,1]\
                     +2*Fo*dx**2*(Tc[-3,0]+Tc[-1,0])+(dx*dy\
                     -2*Fo*(dy**2+2*dx**2)-2*Fo*Bi*dy**2)*Tc[-2,0])/(dx*dy)
        # Apply BC for largest x
        for i in range(len(BC_info['BCx2'])/3):
            # Flux
            if BC_info['BCx2'][3*i]=='F':
                st=BC_info['BCx2'][2+3*i][0]
                en=BC_info['BCx2'][2+3*i][1]
                q=BC_info['BCx2'][1+3*i]
                T[st:en,-1]=(2*Fo*q*dy**2*dx/k+2*Fo*dy**2*Tc[st:en,-2]\
                 +Fo*dx**2*(Tc[st-1:en-1,-1]+Tc[st+1:en+1,-1])+(dx*dy\
                 -2*Fo*(dy**2+dx**2))*Tc[st:en,-1])/(dx*dy)
                if len(BC_info['BCx2'])/3-i==1:
                    T[-2,-1]=(2*Fo*q*dy**2*dx/k+2*Fo*dy**2*Tc[-2,-2]\
                     +Fo*dx**2*(Tc[-3,-1]+Tc[-1,-1])+(dx*dy\
                     -2*Fo*(dy**2+dx**2))*Tc[-2,-1])/(dx*dy)
            # Convective
            if BC_info['BCx2'][3*i]=='C':
                st=BC_info['BCx2'][2+3*i][0]
                en=BC_info['BCx2'][2+3*i][1]
                Bi=BC_info['BCx2'][1+3*i][0]*dx/k
                T[st:en,-1]=(2*Fo*Bi*dy**2*BC_info['BCx2'][1+3*i][1]+2*Fo*dy**2*Tc[st:en,-2]\
                 +2*Fo*dx**2*(Tc[st-1:en-1,-1]+Tc[st+1:en+1,-1])+(dx*dy\
                 -2*Fo*(dy**2+2*dx**2)-2*Fo*Bi*dy**2)*Tc[st:en,0])/(dx*dy)
                if len(BC_info['BCx2'])/3-i==1:
                    T[-2,-1]=(2*Fo*Bi*dy**2*BC_info['BCx2'][1+3*i][1]+2*Fo*dy**2*Tc[-2,-2]\
                     +2*Fo*dx**2*(Tc[-3,-1]+Tc[-1,-1])+(dx*dy\
                     -2*Fo*(dy**2+2*dx**2)-2*Fo*Bi*dy**2)*Tc[-2,0])/(dx*dy)
        # Apply BC for lowest y
        for i in range(len(BC_info['BCy1'])/3):
            # Flux
            if BC_info['BCy1'][3*i]=='F':
                st=BC_info['BCy1'][2+3*i][0]
                en=BC_info['BCy1'][2+3*i][1]
                q=BC_info['BCy1'][1+3*i]
                T[0,st:en]=(2*Fo*q*dx**2*dy/k+2*Fo*dx**2*Tc[1,st:en]\
                 +Fo*dy**2*(Tc[0,st-1:en-1]+Tc[0,st+1:en+1])+(dx*dy\
                 -2*Fo*(dx**2+dy**2))*Tc[0,st:en])/(dx*dy)
                if len(BC_info['BCy1'])/3-i==1:
                    T[0,-2]=(2*Fo*q*dx**2*dy/k+2*Fo*dx**2*Tc[1,-2]\
                     +Fo*dy**2*(Tc[0,-3]+Tc[0,-1])+(dx*dy\
                     -2*Fo*(dx**2+dy**2))*Tc[0,-2])/(dx*dy)
            # Convective
            if BC_info['BCy1'][3*i]=='C':
                st=BC_info['BCy1'][2+3*i][0]
                en=BC_info['BCy1'][2+3*i][1]
                Bi=BC_info['BCy1'][1+3*i][0]*dy/k
                T[0,st:en]=(2*Fo*Bi*dx**2*BC_info['BCy1'][1+3*i][1]+2*Fo*dx**2*Tc[1,st:en]\
                 +2*Fo*dy**2*(Tc[0,st-1:en-1]+Tc[0,st+1:en+1])+(dx*dy\
                 -2*Fo*(dx**2+2*dy**2)-2*Fo*Bi*dx**2)*Tc[0,st:en])/(dx*dy)
                if len(BC_info['BCy1'])/3-i==1:
                    T[0,-2]=(2*Fo*Bi*dx**2*BC_info['BCy1'][1+3*i][1]+2*Fo*dx**2*Tc[1,-2]\
                     +2*Fo*dy**2*(Tc[0,-3]+Tc[0,-1])+(dx*dy\
                     -2*Fo*(dx**2+2*dy**2)-2*Fo*Bi*dx**2)*Tc[0,-2])/(dx*dy)
        # Apply BC for largest y
        for i in range(len(BC_info['BCy2'])/3):
            # Flux
            if BC_info['BCy2'][3*i]=='F':
                st=BC_info['BCy2'][2+3*i][0]
                en=BC_info['BCy2'][2+3*i][1]
                q=BC_info['BCy2'][1+3*i]
                T[-1,st:en]=(2*Fo*q*dx**2*dy/k+2*Fo*dx**2*Tc[-2,st:en]\
                 +Fo*dy**2*(Tc[-1,st-1:en-1]+Tc[-1,st+1:en+1])+(dx*dy\
                 -2*Fo*(dx**2+dy**2))*Tc[-1,st:en])/(dx*dy)
                if len(BC_info['BCy2'])/3-i==1:
                    T[-1,-2]=(2*Fo*q*dx**2*dy/k+2*Fo*dx**2*Tc[-2,-2]\
                     +Fo*dy**2*(Tc[-1,-3]+Tc[-1,-1])+(dx*dy\
                     -2*Fo*(dx**2+dy**2))*Tc[-1,-2])/(dx*dy)
            # Convective
            if BC_info['BCy2'][3*i]=='C':
                st=BC_info['BCy2'][2+3*i][0]
                en=BC_info['BCy2'][2+3*i][1]
                Bi=BC_info['BCy2'][1+3*i][0]*dy/k
                T[-1,st:en]=(2*Fo*Bi*dx**2*BC_info['BCy2'][1+3*i][1]+2*Fo*dx**2*Tc[-2,st:en]\
                 +2*Fo*dy**2*(Tc[-1,st-1:en-1]+Tc[-1,st+1:en+1])+(dx*dy\
                 -2*Fo*(dx**2+2*dy**2)-2*Fo*Bi*dx**2)*Tc[-1,st:en])/(dx*dy)
                if len(BC_info['BCy2'])/3-i==1:
                    T[-1,-2]=(2*Fo*Bi*dx**2*BC_info['BCy2'][1+3*i][1]+2*Fo*dx**2*Tc[-2,-2]\
                     +2*Fo*dy**2*(Tc[-1,-3]+Tc[-1,-1])+(dx*dy\
                     -2*Fo*(dx**2+2*dy**2)-2*Fo*Bi*dx**2)*Tc[-1,-2])/(dx*dy)
    # Implicit solver and flux/convective BCs
    else:
        Tc=T.copy()
        Tprev=T.copy()
        print 'Transient, implicit solver'
        print 'Residuals:'
        while (diff>conv) and (count<1000):
            Tc[1:-1, 1:-1]=(Fo*dx**2*(T[:-2,1:-1]+T[2:,1:-1]) \
            +Fo*dy**2*(T[1:-1,:-2]+T[1:-1,2:])+dx*dy*Tprev[1:-1,1:-1]) \
            /(dx*dy+2*Fo*(dx**2+dy**2))
            Tc2[1:-1, 1:-1]=alpha*Tc[1:-1, 1:-1]+(1-alpha)*T[1:-1, 1:-1]
            
            # Apply lowest x boundary conditions
            for i in range(len(BC_info['BCx1'])/3):
                # Flux
                if BC_info['BCx1'][3*i]=='F':
                    st=BC_info['BCx1'][2+3*i][0]
                    en=BC_info['BCx1'][2+3*i][1]
                    q=BC_info['BCx1'][1+3*i]
                    Tc2[st:en,0]=(2*Fo*q*dy**2*dx/k+2*Fo*dy**2*Tc2[st:en,1]\
                     +Fo*dx**2*(Tc2[st-1:en-1,0]+Tc2[st+1:en+1,0])+dx*dy*Tprev[st:en,0])\
                     /(dx*dy+2*Fo*(dy**2+dx**2))
                    if len(BC_info['BCx1'])/3-i==1:
                         Tc2[-2,0]=(2*Fo*q*dy**2*dx/k+2*Fo*dy**2*Tc2[-2,1]\
                         +Fo*dx**2*(Tc2[-3,0]+Tc2[-1,0])+dx*dy*Tprev[-2,0])\
                         /(dx*dy+2*Fo*(dy**2+dx**2))
                # Convective
                elif BC_info['BCx1'][3*i]=='C':
                    st=BC_info['BCx1'][2+3*i][0]
                    en=BC_info['BCx1'][2+3*i][1]
                    Bi=BC_info['BCx1'][1+3*i][0]*dx/k
                    Tc2[st:en,0]=(2*Fo*Bi*BC_info['BCx1'][1+3*i][1]*dy**2+2*Fo*dy**2*Tc2[st:en,1]\
                     +Fo*dx**2*(Tc2[st-1:en-1,0]+Tc2[st+1:en+1,0])+dx*dy*Tprev[st:en,0])\
                     /(dx*dy+2*Fo*(dy**2+dx**2+Bi*dy**2))
                    if len(BC_info['BCx1'])/3-i==1:
                        Tc2[-2,0]=(2*Fo*Bi*BC_info['BCx1'][1+3*i][1]*dy**2+2*Fo*dy**2*Tc2[-2,1]\
                         +Fo*dx**2*(Tc2[-3,0]+Tc2[-1,0])+dx*dy*Tprev[-2,0])\
                         /(dx*dy+2*Fo*(dy**2+dx**2+Bi*dy**2))
            # Apply largest x boundary conditions
            for i in range(len(BC_info['BCx2'])/3):
                # Flux
                if BC_info['BCx2'][3*i]=='F':
                    st=BC_info['BCx2'][2+3*i][0]
                    en=BC_info['BCx2'][2+3*i][1]
                    q=BC_info['BCx2'][1+3*i]
                    Tc2[st:en,-1]=(2*Fo*q*dy**2*dx/k+2*Fo*dy**2*Tc2[st:en,-2]\
                     +Fo*dx**2*(Tc2[st-1:en-1,-1]+Tc2[st+1:en+1,-1])+dx*dy*Tprev[st:en,-1])\
                     /(dx*dy+2*Fo*(dy**2+dx**2))
                    if len(BC_info['BCx2'])/3-i==1:
                        Tc2[-2,-1]=(2*Fo*q*dy**2*dx/k+2*Fo*dy**2*Tc2[-2,-2]\
                         +Fo*dx**2*(Tc2[-3,-1]+Tc2[-1,-1])+dx*dy*Tprev[-2,-1])\
                         /(dx*dy+2*Fo*(dy**2+dx**2))
                # Convective
                if BC_info['BCx2'][3*i]=='C':
                    st=BC_info['BCx2'][2+3*i][0]
                    en=BC_info['BCx2'][2+3*i][1]
                    Bi=BC_info['BCx2'][1+3*i][0]*dx/k
                    Tc2[st:en,-1]=(2*Fo*Bi*BC_info['BCx2'][1+3*i][1]*dy**2+2*Fo*dy**2*Tc2[st:en,-2]\
                     +Fo*dx**2*(Tc2[st-1:en-1,-1]+Tc2[st+1:en+1,-1])+dx*dy*Tprev[st:en,-1])\
                     /(dx*dy+2*Fo*(dy**2+dx**2+Bi*dy**2))
                    if len(BC_info['BCx2'])/3-i==1:
                        Tc2[-2,-1]=(2*Fo*Bi*BC_info['BCx2'][1+3*i][1]*dy**2+2*Fo*dy**2*Tc2[-2,-2]\
                         +Fo*dx**2*(Tc2[-3,-1]+Tc2[-1,-1])+dx*dy*Tprev[-2,-1])\
                         /(dx*dy+2*Fo*(dy**2+dx**2+Bi*dy**2))
            # Apply lowest y boundary conditions
            for i in range(len(BC_info['BCy1'])/3):
                # Flux
                if BC_info['BCy1'][3*i]=='F':
                    st=BC_info['BCy1'][2+3*i][0]
                    en=BC_info['BCy1'][2+3*i][1]
                    q=BC_info['BCy1'][1+3*i]
                    Tc2[0,st:en]=(2*Fo*q*dx**2*dy/k+2*Fo*dx**2*Tc2[1,st:en]\
                     +Fo*dy**2*(Tc2[0,st-1:en-1]+Tc2[0,st+1:en+1])+dx*dy*Tprev[0,st:en])\
                     /(dx*dy+2*Fo*(dy**2+dx**2))
                    if len(BC_info['BCy1'])/3-i==1:
                        Tc2[0,-2]=(2*Fo*q*dx**2*dy/k+2*Fo*dx**2*Tc2[1,-2]\
                         +Fo*dy**2*(Tc2[0,-3]+Tc2[0,-1])+dx*dy*Tprev[0,-2])\
                         /(dx*dy+2*Fo*(dy**2+dx**2))
                # Convective
                if BC_info['BCy1'][3*i]=='C':
                    st=BC_info['BCy1'][2+3*i][0]
                    en=BC_info['BCy1'][2+3*i][1]
                    Bi=BC_info['BCy1'][1+3*i][0]*dy/k
                    Tc2[0,st:en]=(2*Fo*Bi*BC_info['BCy1'][1+3*i][1]*dx**2+2*Fo*dx**2*Tc2[1,st:en]\
                     +Fo*dy**2*(Tc2[0,st-1:en-1]+Tc2[0,st+1:en+1])+dx*dy*Tprev[0,st:en])\
                     /(dx*dy+2*Fo*(dy**2+dx**2+Bi*dx**2))
                    if len(BC_info['BCy1'])/3-i==1:
                        Tc2[0,-2]=(2*Fo*Bi*BC_info['BCy1'][1+3*i][1]*dx**2+2*Fo*dx**2*Tc2[1,-2]\
                         +Fo*dy**2*(Tc2[0,-3]+Tc2[0,-1])+dx*dy*Tprev[0,-2])\
                         /(dx*dy+2*Fo*(dy**2+dx**2+Bi*dx**2))
            # Apply largest y boundary conditions
            for i in range(len(BC_info['BCy2'])/3):
                # Flux
                if BC_info['BCy2'][3*i]=='F':
                    st=BC_info['BCy2'][2+3*i][0]
                    en=BC_info['BCy2'][2+3*i][1]
                    q=BC_info['BCy2'][1+3*i]
                    Tc2[-1,st:en]=(2*Fo*q*dx**2*dy/k+2*Fo*dx**2*Tc2[-2,st:en]\
                     +Fo*dy**2*(Tc2[-1,st-1:en-1]+Tc2[-1,st+1:en+1])+dx*dy*Tprev[-1,st:en])\
                     /(dx*dy+2*Fo*(dy**2+dx**2))
                    if len(BC_info['BCy2'])/3-i==1:
                        Tc2[-1,-2]=(2*Fo*q*dx**2*dy/k+2*Fo*dx**2*Tc2[-2,-2]\
                         +Fo*dy**2*(Tc2[-1,-3]+Tc2[-1,-1])+dx*dy*Tprev[-1,-2])\
                         /(dx*dy+2*Fo*(dy**2+dx**2))
                # Convective
                if BC_info['BCy2'][3*i]=='C':
                    st=BC_info['BCy2'][2+3*i][0]
                    en=BC_info['BCy2'][2+3*i][1]
                    Bi=BC_info['BCy2'][1+3*i][0]*dy/k
                    Tc2[-1,st:en]=(2*Fo*Bi*BC_info['BCy2'][1+3*i][1]*dx**2+2*Fo*dx**2*Tc2[-2,st:en]\
                     +Fo*dy**2*(Tc2[-1,st-1:en-1]+Tc2[-1,st+1:en+1])+dx*dy*Tprev[-1,st:en])\
                     /(dx*dy+2*Fo*(dy**2+dx**2+Bi*dx**2))
                    if len(BC_info['BCy1'])/3-i==1:
                        Tc2[-1,-2]=(2*Fo*Bi*BC_info['BCy2'][1+3*i][1]*dx**2+2*Fo*dx**2*Tc2[-2,-2]\
                         +Fo*dy**2*(Tc2[-1,-3]+Tc2[-1,-1])+dx*dy*Tprev[-1,-2])\
                         /(dx*dy+2*Fo*(dy**2+dx**2+Bi*dx**2))
            
            diff=numpy.sum(numpy.abs(T[:]-Tc2[:]))/numpy.sum(numpy.abs(T[:]))
            count=count+1
            print(diff)
            T=Tc2.copy()
    if count==1000:
        error=1
    return T, error, Fo
    
# 3D plotter
def PlotXYT(X, Y, T, T_lower_lim, T_upper_lim,title):
    fig = pyplot.figure(figsize=(7, 7), dpi=100)
    ax = fig.gca(projection='3d')
    ax.plot_surface(X*1000, Y*1000, T[:], rstride=1, cstride=1, cmap=cm.viridis,linewidth=0, antialiased=True)
    ax.set_zlim(T_lower_lim, T_upper_lim)
    ax.set_xlabel('$x$ (mm)')
    ax.set_ylabel('$y$ (mm)')
    ax.set_zlabel('T (K)')
    pyplot.title(title)

# ------------------Setup-----------------------------------
L=10**(-3) # Length of plate (in x direction)
W=6*10**(-3) # Width of plate (in y direction)
Nx=101 # Number of nodes across length
Ny=601 # Number of nodes across width
k=10 # Thermal conductivity (W/m/K)
rho=8000 # Density (kg/m^3)
Cp=800 # Specific heat (J/kg/K)

dx=L/(Nx-1)
dy=W/(Ny-1)
T=numpy.zeros((Ny, Nx))

#       Convergence
conv=.0001 # Convergence target (SS and implicit trans solvers)
dt=10**(-7)
#dt=0.1*rho*Cp*dx*dy/k # time step
timeSteps=1000 # number of time steps (transient)
alpha=1 # Relaxation parameter (<1-under, >1-over)

#       Initial conditions
T[:, :]=300

#       BCs on ends of length
# Format: 'BCx1': (type,info,range,...)
# BC_type: 'T'-temp, 'F'-flux, 'C'-convective
# BC_info: temp/flux value OR h then Tinf (convective BC)
#range: (0,-1)-whole boundary (temp), (1,-2) whole boundary (flux), (0, 2)-node numbers

#BC_info={'BCy1': ('T',600,(0,-1)),\
#         'BCy2': ('T',300,(0,-1)),\
#         'BCx1': ('C',(50,300),(1,-2)),\
#         'BCx2': ('C',(50,300),(1,-2))\
#         }

BC_info={'BCx1': ('F',0,(1,-2)),\
         'BCx2': ('C',(50,300),(1,-2)),\
         'BCy1': ('F',0,(1,-2)),\
         'BCy2': ('F',0.6*4*10**8,(1,int(0.05*10**(-3)/dy+1)),'C',(50,300),(int(0.05*10**(-3)/dy+1),-2))\
         }
# ----------------Solve and Plot (uncomment desired settings)
x=numpy.linspace(0, L, Nx)
y=numpy.linspace(0, W, Ny)
X, Y = numpy.meshgrid(x, y)
Fo=dt/rho/Cp/dx/dy*k

#       Steady state solver/plotter
#T,error=SteadySolve(T, (dx,dy), k, (conv,alpha),BC_info)
#PlotXYT(X, Y, T, 300, 1000,'Time step 0')

#       Transient solver
for i in range(timeSteps):
    # Change a BC with time
#    if i%2==0:
#        Tx1=Tx1-50
    print 'Time step %i'%i
    T,error,Fo=TransSolve(1, T, (dx,dy), k, Fo, (conv,alpha),BC_info)

    if error==1:
        print 'Convergence problem at time step %i'%i
        break

#    if i==timeSteps/2:
#        PlotXYT(X,Y,T,300,800,'Time step %i'%i)

PlotXYT(X, Y, T, 300, 800,'Time step %i'%i)
#plotx=dx
##int(plotx/dx)
#pyplot.figure()
#pyplot.plot(Y[:,int(plotx/dx)]*1000, T[:,int(plotx/dx)])
#pyplot.xlabel('Y (mm)')
#pyplot.ylabel('T (K)')
#pyplot.xlim([4,6])
#pyplot.title('Temperature distribution at X=%3f mm'%(plotx*1000))

#BC_info['BCy2']=('C',(50,300),(1,-2))
#for i in range(timeSteps):
#    # Change a BC with time
##    if i%2==0:
##        Tx1=Tx1-50
#    print 'Time step %i'%i
#    T,error,Fo=TransSolve(1, T, (dx,dy), k, Fo, (conv,alpha),BC_info)
#
#    if error==1:
#        print 'Convergence problem at time step %i'%i
#        break
#
##    if i==timeSteps/2:
##        PlotXYT(X,Y,T,300,800,'Time step %i'%i)
# 2D plots
#ploty=2
#pyplot.plot(X[1,:], T[int(ploty/dy),:])
#pyplot.xlabel('X')
#pyplot.ylabel('T')
#pyplot.xlim([0,1])
##pyplot.title('Temperature distribution')
#
plotx=dx
#int(plotx/dx)
pyplot.figure()
pyplot.plot(Y[:,int(plotx/dx)]*1000, T[:,int(plotx/dx)])
pyplot.xlabel('Y (mm)')
pyplot.ylabel('T (K)')
pyplot.xlim([4,6])
pyplot.title('Temperature distribution at X=%3f mm'%(plotx*1000))