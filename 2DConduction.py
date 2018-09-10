# -*- coding: utf-8 -*-
"""
Started on Thu Sep 06 20:09:37 2018

@author: Joseph

This script contains solvers for 2D planar conduction:

Steady, transient (explicit)
with temperature, flux BCs, convective (any number of)
NEED TO DEAL WITH CORNERS WITH 2 FLUX/CONVECTIVE BCS

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
    is_x_BC: (bool) indicates whether applied BC is at length ends e.g. (2,:)
    is_xy_end: (bool) indicates BC is applied to last coordinate of that dimension
    type_BC: designates BC type; 0-temps, 1-flux, 2-convective
    
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
def SteadySolve(T, dxy, k, conv_param, BC_setup, bc1, bc2, bc3, bc4):
    dx,dy=dxy
    conv,alpha=conv_param
    error=0
    diff=10
    count=1
    # Assign temperature BCs if applicable
    if BC_setup[0]==1:
        T[:,0]=bc1
    if BC_setup[1]==1:
        T[:,-1]=bc2
    if BC_setup[2]==1:
        T[0,:]=bc3
    if BC_setup[3]==1:
        T[-1,:]=bc4
        
    Tc2=T.copy()
    print 'Residuals:'
    while (diff>conv) and (count<1000):
        Tc=T.copy()
        Tc2[1:-1, 1:-1]=(dx**2*(Tc[:-2,1:-1]+Tc[2:,1:-1]) \
        +dy**2*(Tc[1:-1,:-2]+Tc[1:-1,2:])) \
        /(2*dx**2+2*dy**2)
        T[1:-1, 1:-1]=alpha*Tc2[1:-1, 1:-1]+(1-alpha)*Tc[1:-1, 1:-1]
        
        # Apply flux BC if applicable
        if BC_setup[0]==2:
            T[1:-1,0]=(2*bc1*dy**2*dx/k+2*dy**2*T[1:-1,1]\
             +dx**2*(T[:-2,0]+T[2:,0]))/(2*dy**2+2*dx**2)
        if BC_setup[1]==2:
            T[1:-1,-1]=(2*bc2*dy**2*dx/k+2*dy**2*T[1:-1,-2]\
             +dx**2*(T[:-2,-1]+T[2:,-1]))/(2*dy**2+2*dx**2)
        if BC_setup[2]==2:
            T[0,1:-1]=(2*bc3*dx**2*dy/k+2*dx**2*T[1,1:-1]\
            +dy**2*(T[0,:-2]+T[0,2:]))/(2*dx**2+2*dy**2)
        if BC_setup[3]==2:
            T[-1,1:-1]=(2*bc4*dx**2*dy/k+2*dx**2*T[-2,1:-1]\
            +dy**2*(T[-1,:-2]+T[-1,2:]))/(2*dx**2+2*dy**2)
        
        # Apply convective BC if applicable
        if BC_setup[0]==3:
            Bi=bc1[0]*dx/k
            T[1:-1,0]=(2*Bi*dy**2*bc1[1]+2*dy**2*T[1:-1,1]\
             +dx**2*(T[:-2,0]+T[2:,0]))/(2*dy**2+2*dx**2+2*Bi*dy**2)
        if BC_setup[1]==3:
            Bi=bc2[0]*dx/k
            T[1:-1,-1]=(2*Bi*dy**2*bc2[1]+2*dy**2*T[1:-1,-2]\
             +dx**2*(T[:-2,-1]+T[2:,-1]))/(2*dy**2+2*dx**2+2*Bi*dy**2)
        if BC_setup[2]==3:
            Bi=bc3[0]*dy/k
            T[0,1:-1]=(2*Bi*dx**2*bc3[1]+2*dx**2*T[1,1:-1]\
            +dy**2*(T[0,:-2]+T[0,2:]))/(2*dx**2+2*dy**2+2*Bi*dx**2)
        if BC_setup[3]==3:
            Bi=bc4[0]*dy/k
            T[-1,1:-1]=(2*Bi*dx**2*bc4[1]+2*dx**2*T[-2,1:-1]\
            +dy**2*(T[-1,:-2]+T[-1,2:]))/(2*dx**2+2*dy**2+2*Bi*dx**2)
        
        diff=numpy.sum(numpy.abs(T[:]-Tc[:]))/numpy.sum(numpy.abs(T[:]))
        count=count+1
        print(diff)
    
    if count==1000:
        print 'Convergence problem'
        error=1
    return T, error
    
# Transient solvers
def TransSolve(is_explicit, T, dxy, k, Fo, conv_param, BC_setup, bc1, bc2, bc3, bc4):
    dx,dy=dxy
    conv,alpha=conv_param
    error=0
    diff=10
    count=1
    # Assign temperature BCs if applicable
    if BC_setup[0]==1:
        T[:,0]=bc1
    if BC_setup[1]==1:
        T[:,-1]=bc2
    if BC_setup[2]==1:
        T[0,:]=bc3
    if BC_setup[3]==1:
        T[-1,:]=bc4
    
    Tc2=T.copy()
    
    if is_explicit:
        Tc=T.copy()
        Tc2[1:-1,1:-1]=(Fo*dy**2*(Tc[1:-1,:-2]+Tc[1:-1,2:])\
            +Fo*dx**2*(Tc[:-2,1:-1]+Tc[2:,1:-1])\
            +(dx*dy-2*Fo*(dx**2+dy**2))*Tc[1:-1,1:-1])/(dx*dy)
        T[1:-1, 1:-1]=alpha*Tc2[1:-1, 1:-1]+(1-alpha)*Tc[1:-1, 1:-1]
        
        # Apply flux BC if applicable
        if BC_setup[0]==2:
            T[1:-1,0]=(2*Fo*bc1*dy**2*dx/k+2*Fo*dy**2*Tc[1:-1,1]\
             +2*Fo*dx**2*(Tc[:-2,0]+Tc[2:,0])+(dx*dy\
             -2*Fo*(dy**2+2*dx**2)*Tc[1:-1,0]))/(dx*dy)
        if BC_setup[1]==2:
            T[1:-1,-1]=(2*Fo*bc2*dy**2*dx/k+2*Fo*dy**2*Tc[1:-1,-2]\
             +2*Fo*dx**2*(Tc[:-2,-1]+Tc[2:,-1])+(dx*dy\
             -2*Fo*(dy**2+2*dx**2)*Tc[1:-1,-1]))/(dx*dy)
        if BC_setup[2]==2:
            T[0,1:-1]=(2*Fo*bc3*dx**2*dy/k+2*Fo*dx**2*Tc[1,1:-1]\
             +2*Fo*dy**2*(Tc[0,:-2]+Tc[0,2:])+(dx*dy\
             -2*Fo*(dx**2+2*dy**2)*Tc[0,1:-1]))/(dx*dy)
        if BC_setup[3]==2:
            T[-1,1:-1]=(2*Fo*bc3*dx**2*dy/k+2*Fo*dx**2*Tc[-2,1:-1]\
             +2*Fo*dy**2*(Tc[-1,:-2]+Tc[-1,2:])+(dx*dy\
             -2*Fo*(dx**2+2*dy**2)*Tc[-1,1:-1]))/(dx*dy)
        
        # Apply convective BC if applicable
        if BC_setup[0]==3:
            Bi=bc1[0]*dx/k
            T[1:-1,0]=(2*Fo*Bi*dy**2*bc1[1]+2*Fo*dy**2*Tc[1:-1,1]\
             +2*Fo*dx**2*(Tc[:-2,0]+Tc[2:,0])+(dx*dy\
             -2*Fo*(dy**2+2*dx**2)-2*Fo*Bi*dy**2)*Tc[1:-1,0])/(dx*dy)
        if BC_setup[1]==3:
            Bi=bc2[0]*dx/k
            T[1:-1,-1]=(2*Fo*Bi*dy**2*bc2[1]+2*Fo*dy**2*Tc[1:-1,-2]\
             +2*Fo*dx**2*(Tc[:-2,-1]+Tc[2:,-1])+(dx*dy\
             -2*Fo*(dy**2+2*dx**2)-2*Fo*Bi*dy**2)*Tc[1:-1,0])/(dx*dy)
        if BC_setup[2]==3:
            Bi=bc3[0]*dy/k
            T[0,1:-1]=(2*Fo*Bi*dx**2*bc3[1]+2*Fo*dx**2*Tc[1,1:-1]\
             +2*Fo*dy**2*(Tc[0,:-2]+Tc[0,2:])+(dx*dy\
             -2*Fo*(dx**2+2*dy**2)-2*Fo*Bi*dx**2)*Tc[0,1:-1])/(dx*dy)
        if BC_setup[3]==3:
            Bi=bc4[0]*dy/k
            T[-1,1:-1]=(2*Fo*Bi*dx**2*bc4[1]+2*Fo*dx**2*Tc[-2,1:-1]\
             +2*Fo*dy**2*(Tc[-1,:-2]+Tc[-1,2:])+(dx*dy\
             -2*Fo*(dx**2+2*dy**2)-2*Fo*Bi*dx**2)*Tc[-1,1:-1])/(dx*dy)
    
    else:
        Tc=T.copy()
        Tprev=T.copy()
        print 'Residuals:'
        while (diff>conv) and (count<1000):
            Tc[1:-1, 1:-1]=(Fo*dx**2*(T[:-2,1:-1]+T[2:,1:-1]) \
            +Fo*dy**2*(T[1:-1,:-2]+T[1:-1,2:])+dx*dy*Tprev[1:-1,1:-1]) \
            /(dx*dy+2*Fo*(dx**2+dy**2))
            Tc2[1:-1, 1:-1]=alpha*Tc[1:-1, 1:-1]+(1-alpha)*T[1:-1, 1:-1]
            
            # Apply flux BC if applicable (FILL IN)
            if BC_setup[0]==2:
                T[1:-1,0]=(2*bc1*dy**2*dx/k+2*dy**2*T[1:-1,1]\
                 +dx**2*(T[:-2,0]+T[2:,0]))/(2*dy**2+2*dx**2)
            if BC_setup[1]==2:
                T[1:-1,-1]=(2*bc2*dy**2*dx/k+2*dy**2*T[1:-1,-2]\
                 +dx**2*(T[:-2,-1]+T[2:,-1]))/(2*dy**2+2*dx**2)
            if BC_setup[2]==2:
                T[0,1:-1]=(2*bc3*dx**2*dy/k+2*dx**2*T[1,1:-1]\
                +dy**2*(T[0,:-2]+T[0,2:]))/(2*dx**2+2*dy**2)
            if BC_setup[3]==2:
                T[-1,1:-1]=(2*bc4*dx**2*dy/k+2*dx**2*T[-2,1:-1]\
                +dy**2*(T[-1,:-2]+T[-1,2:]))/(2*dx**2+2*dy**2)
            
            # Apply convective BC if applicable (FILL IN)
            if BC_setup[0]==3:
                Bi=bc1[0]*dx/k
                T[1:-1,0]=(2*Bi*dy**2*bc1[1]+2*dy**2*T[1:-1,1]\
                 +dx**2*(T[:-2,0]+T[2:,0]))/(2*dy**2+2*dx**2+2*Bi*dy**2)
            if BC_setup[1]==3:
                Bi=bc2[0]*dx/k
                T[1:-1,-1]=(2*Bi*dy**2*bc2[1]+2*dy**2*T[1:-1,-2]\
                 +dx**2*(T[:-2,-1]+T[2:,-1]))/(2*dy**2+2*dx**2+2*Bi*dy**2)
            if BC_setup[2]==3:
                Bi=bc3[0]*dy/k
                T[0,1:-1]=(2*Bi*dx**2*bc3[1]+2*dx**2*T[1,1:-1]\
                +dy**2*(T[0,:-2]+T[0,2:]))/(2*dx**2+2*dy**2+2*Bi*dx**2)
            if BC_setup[3]==3:
                Bi=bc4[0]*dy/k
                T[-1,1:-1]=(2*Bi*dx**2*bc4[1]+2*dx**2*T[-2,1:-1]\
                +dy**2*(T[-1,:-2]+T[-1,2:]))/(2*dx**2+2*dy**2+2*Bi*dx**2)
            
            diff=numpy.sum(numpy.abs(T[:]-Tc2[:]))/numpy.sum(numpy.abs(T[:]))
            count=count+1
            print(diff)
            T=Tc2.copy()
    if count==1000:
        print 'Convergence problem'
        error=1
    return T, error
    
# 3D plotter
def PlotXYT(X, Y, T, T_lower_lim, T_upper_lim):
    fig = pyplot.figure(figsize=(7, 7), dpi=100)
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, T[:], rstride=1, cstride=1, cmap=cm.viridis,linewidth=0, antialiased=True)
    ax.set_zlim(T_lower_lim, T_upper_lim)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('T')

# ------------------Setup-----------------------------------
L=4.0 # Length of plate (in x direction)
W=4.0 # Width of plate (in y direction)
Nx=10 # Number of nodes across length
Ny=10 # Number of nodes across width
k=25 # Thermal conductivity (W/m/K)
rho=8000 # Density (kg/m^3)
Cp=800 # Specific heat (J/kg/K)
Fo=0.1 # Fourier number (less than 0.25 for explicit method)

dx=L/(Nx-1)
dy=W/(Ny-1)
dt=Fo*rho*Cp*dx*dy/k
T=numpy.zeros((Ny, Nx))

#       Convergence
conv=.001 # Convergence target (SS and implicit trans solvers)
timeSteps=10 # number of time steps (transient)
TranConvStab=0.2 # Fo(Bi+1)<=0.25? (trans, convective BC)
alpha=1 # Relaxation parameter (<1-under, >1-over)

#       Initial conditions
T[1:-1, 1:-1]=600

#       BCs on ends of length
Tx1=700 #                        SMALLEST x coordinate
qx1=1000 # Heat flux BC
#hx1=50 # Convective heat transfer coefficient (W/m^2/K)
#Tinfx1=273 # Freestream temperature
Tx2=300 #                       LARGEST x coordinate
qx2=1000 # Heat flux BC
#hx2=50 # Convective heat transfer coefficient (W/m^2/K)
#Tinfx2=273 # Freestream temperature

#       BCs on ends of width
Ty1=300 #                        SMALLEST y coordinate
qy1=1000 # Heat flux BC
#hy1=50 # Convective heat transfer coefficient (W/m^2/K)
#Tinfy1=273 # Freestream temperature
Ty2=700 #                        LARGEST y coordinate
qy2=1000 # Heat flux BC
#hy2=50 # Convective heat transfer coefficient (W/m^2/K)
#Tinfy2=273 # Freestream temperature

# ----------------Solve and Plot (uncomment desired settings)
x=numpy.linspace(0, L, Nx)
y=numpy.linspace(0, W, Ny)
X, Y = numpy.meshgrid(x, y)

#T,error=SteadySolve(T, (dx,dy), k, (conv,alpha), \
#                    (1,1,1,1), Tx1, Tx2, Ty1, Ty2)
for i in range(timeSteps):
    if i%2==0:
        Tx1=Tx1-50
    T,error=TransSolve(1, T, (dx,dy), k, Fo, (conv,alpha),\
                       (1,1,1,1), Tx1, Tx2, Ty1, Ty2)
    if i==timeSteps/2:
        PlotXYT(X,Y,T,300,700)
#T,error=TransSolve(0, T, (dx,dy), k, Fo, (conv,alpha), \
#                   (1,1,1,1), Tx1, Tx2, Ty1, Ty2)

PlotXYT(X, Y, T, 300, 700)

#print 'Transient model (implicit) with convective and temperature BCs. dt=%.2fs for %i timesteps\nResiduals:'%(dt,timeSteps)
#for i in range(timeSteps):
#    T,error=TransSolve(1, T, (dx,dy), k, Fo, (conv,alpha), (1,1,1,1), 700, 300, 300, 700)
#    T,error=TransSolve(0, T, (dx,dy), k, Fo, (conv,alpha), (1,1,1,1), 700, 300, 300, 700)
#PlotXYT(X, Y, T, 300, 700)

