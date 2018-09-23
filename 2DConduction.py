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
    print 'Steady state solver'
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
        error=1
    return T, error
    
# Transient solvers
def TransSolve(is_explicit, T, dxy, k, Fo, conv_param, BC_setup, bc1, bc2, bc3, bc4):
    dx,dy=dxy
    conv,alpha=conv_param
    error=0
    diff=10
    count=1
    Fo_old=Fo
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
    # Explicit solver and flux/convective BCs
    if is_explicit:
        # Apply stability criteria to Fo (temp BCs)
        Fo=min(Fo, dx*dy/(2*(dy**2+dx**2)))
        # Apply stability criteria to Fo (convective BCs)
        if (BC_setup[0]==3) and (BC_setup[1]==3):
            Bi1=bc1[0]*dx/k
            Bi2=bc2[0]*dx/k
            Fo=min(Fo, 0.5*dx/((dy**2+2*dx**2)/dy+2*Bi1*dy),\
                   0.5*dx/((dy**2+2*dx**2)/dy+2*Bi2*dy))
        elif (BC_setup[0]==3) or (BC_setup[1]==3):
            if BC_setup[0]==3:
                Bi=bc1[0]*dx/k
            else:
                Bi=bc2[0]*dx/k
            Fo=min(Fo, 0.5*dx/((dy**2+2*dx**2)/dy+2*Bi*dy))
        if (BC_setup[2]==3) and (BC_setup[3]==3):
            Bi1=bc3[0]*dy/k
            Bi2=bc4[0]*dy/k
            Fo=min(Fo, 0.5*dy/((dx**2+2*dy**2)/dx+2*Bi1*dx),\
                   0.5*dy/((dx**2+2*dy**2)/dy+2*Bi2*dx))
        elif (BC_setup[2]==3) or (BC_setup[3]==3):
            if BC_setup[2]==3:
                Bi=bc3[0]*dy/k
            else:
                Bi=bc4[0]*dy/k
            Fo=min(Fo, 0.5*dy/((dx**2+2*dy**2)/dy+2*Bi2*dx))
        if Fo!=Fo_old:
            print 'Fourrier number adjusted to %.2f for stability'%Fo
        # Proceed to solve
        Tc=T.copy()
        T[1:-1,1:-1]=(Fo*dy**2*(Tc[1:-1,:-2]+Tc[1:-1,2:])\
            +Fo*dx**2*(Tc[:-2,1:-1]+Tc[2:,1:-1])\
            +(dx*dy-2*Fo*(dx**2+dy**2))*Tc[1:-1,1:-1])/(dx*dy)
        
        #T[1:-1, 1:-1]=alpha*Tc2[1:-1, 1:-1]+(1-alpha)*Tc[1:-1, 1:-1]
        
        # Apply flux BC if applicable
        if BC_setup[0]==2:
            T[1:-1,0]=(2*Fo*bc1*dy**2*dx/k+2*Fo*dy**2*Tc[1:-1,1]\
             +Fo*dx**2*(Tc[:-2,0]+Tc[2:,0])+(dx*dy\
             -2*Fo*(dy**2+dx**2))*Tc[1:-1,0])/(dx*dy)
        if BC_setup[1]==2:
            T[1:-1,-1]=(2*Fo*bc2*dy**2*dx/k+2*Fo*dy**2*Tc[1:-1,-2]\
             +Fo*dx**2*(Tc[:-2,-1]+Tc[2:,-1])+(dx*dy\
             -2*Fo*(dy**2+dx**2))*Tc[1:-1,-1])/(dx*dy)
        if BC_setup[2]==2:
            T[0,1:-1]=(2*Fo*bc3*dx**2*dy/k+2*Fo*dx**2*Tc[1,1:-1]\
             +Fo*dy**2*(Tc[0,:-2]+Tc[0,2:])+(dx*dy\
             -2*Fo*(dx**2+dy**2))*Tc[0,1:-1])/(dx*dy)
        if BC_setup[3]==2:
            T[-1,1:-1]=(2*Fo*bc4*dx**2*dy/k+2*Fo*dx**2*Tc[-2,1:-1]\
             +Fo*dy**2*(Tc[-1,:-2]+Tc[-1,2:])+(dx*dy\
             -2*Fo*(dx**2+dy**2))*Tc[-1,1:-1])/(dx*dy)
        
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
            
            # Apply flux BC if applicable
            if BC_setup[0]==2:
                Tc2[1:-1,0]=(2*Fo*bc1*dy**2*dx/k+2*Fo*dy**2*Tc2[1:-1,1]\
                 +Fo*dx**2*(Tc2[:-2,0]+Tc2[2:,0])+dx*dy*Tprev[1:-1,0])\
                 /(dx*dy+2*Fo*(dy**2+dx**2))
            if BC_setup[1]==2:
                Tc2[1:-1,-1]=(2*Fo*bc2*dy**2*dx/k+2*Fo*dy**2*Tc2[1:-1,-2]\
                 +Fo*dx**2*(Tc2[:-2,-1]+Tc2[2:,-1])+dx*dy*Tprev[1:-1,-1])\
                 /(dx*dy+2*Fo*(dy**2+dx**2))
            if BC_setup[2]==2:
                Tc2[0,1:-1]=(2*Fo*bc3*dx**2*dy/k+2*Fo*dx**2*Tc2[1,1:-1]\
                 +Fo*dy**2*(Tc2[0,:-2]+Tc2[0,2:])+dx*dy*Tprev[0,1:-1])\
                 /(dx*dy+2*Fo*(dy**2+dx**2))
            if BC_setup[3]==2:
                Tc2[-1,1:-1]=(2*Fo*bc4*dx**2*dy/k+2*Fo*dx**2*Tc2[-2,1:-1]\
                 +Fo*dy**2*(Tc2[-1,:-2]+Tc2[-1,2:])+dx*dy*Tprev[-1,1:-1])\
                 /(dx*dy+2*Fo*(dy**2+dx**2))
            
            # Apply convective BC if applicable
            if BC_setup[0]==3:
                Bi=bc1[0]*dx/k
                Tc2[1:-1,0]=(2*Fo*Bi*bc1[1]*dy**2+2*Fo*dy**2*Tc2[1:-1,1]\
                 +Fo*dx**2*(Tc2[:-2,0]+Tc2[2:,0])+dx*dy*Tprev[1:-1,0])\
                 /(dx*dy+2*Fo*(dy**2+dx**2+Bi*dy**2))
            if BC_setup[1]==3:
                Bi=bc2[0]*dx/k
                Tc2[1:-1,-1]=(2*Fo*Bi*bc2[1]*dy**2+2*Fo*dy**2*Tc2[1:-1,-2]\
                 +Fo*dx**2*(Tc2[:-2,-1]+Tc2[2:,-1])+dx*dy*Tprev[1:-1,-1])\
                 /(dx*dy+2*Fo*(dy**2+dx**2+Bi*dy**2))
            if BC_setup[2]==3:
                Bi=bc3[0]*dy/k
                Tc2[0,1:-1]=(2*Fo*Bi*bc3[1]*dx**2+2*Fo*dx**2*Tc2[1,1:-1]\
                 +Fo*dy**2*(Tc2[0,:-2]+Tc2[0,2:])+dx*dy*Tprev[0,1:-1])\
                 /(dx*dy+2*Fo*(dy**2+dx**2+Bi*dx**2))
            if BC_setup[3]==3:
                Bi=bc4[0]*dy/k
                Tc2[-1,1:-1]=(2*Fo*Bi*bc4[1]*dx**2+2*Fo*dx**2*Tc2[-2,1:-1]\
                 +Fo*dy**2*(Tc2[-1,:-2]+Tc2[-1,2:])+dx*dy*Tprev[-1,1:-1])\
                 /(dx*dy+2*Fo*(dy**2+dx**2+Bi*dx**2))
            
            diff=numpy.sum(numpy.abs(T[:]-Tc2[:]))/numpy.sum(numpy.abs(T[:]))
            count=count+1
            print(diff)
            T=Tc2.copy()
    if count==1000:
        error=1
    return T, error, Fo
    
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
Nx=13 # Number of nodes across length
Ny=13 # Number of nodes across width
k=25 # Thermal conductivity (W/m/K)
rho=8000 # Density (kg/m^3)
Cp=800 # Specific heat (J/kg/K)

dx=L/(Nx-1)
dy=W/(Ny-1)
T=numpy.zeros((Ny, Nx))

#       Convergence
conv=.001 # Convergence target (SS and implicit trans solvers)
Fo=0.2 # Fourier number
timeSteps=10 # number of time steps (transient)
alpha=1 # Relaxation parameter (<1-under, >1-over)

#       Initial conditions
T[:, :]=400
dt=Fo*rho*Cp*dx*dy/k

#       BCs on ends of length
Tx1=600 #                         SMALLEST x coordinate
#Tx1=numpy.zeros(Ny) #      VARYING BC
#Tx1[:]=range(300, 600+300/(Ny-1), (300/(Ny-1)))
#qx1=1000 # Heat flux BC
#hx1=50 # Convective heat transfer coefficient (W/m^2/K)
#Tinfx1=300 # Freestream temperature
Tx2=300 #                       LARGEST x coordinate
#Tx2=numpy.zeros(Ny) #      VARYING BC
#qx2=1000 # Heat flux BC
#hx2=50 # Convective heat transfer coefficient (W/m^2/K)
#Tinfx2=300 # Freestream temperature

#       BCs on ends of width
Ty1=600 #                        SMALLEST y coordinate
#Ty1=numpy.zeros(Nx) #      VARYING BC
#qy1=numpy.zeros(timeSteps)
#qy1[:]=range(200, 2000+2000/(timeSteps), 2000/(timeSteps)) # Heat flux BC
qy1=1000
hy1=50 # Convective heat transfer coefficient (W/m^2/K)
Tinfy1=300 # Freestream temperature
Ty2=300 #                        LARGEST y coordinate
#Ty2=numpy.zeros(Nx) #      VARYING BC
qy2=1000 # Heat flux BC
#hy2=50 # Convective heat transfer coefficient (W/m^2/K)
#Tinfy2=300 # Freestream temperature

# ----------------Solve and Plot (uncomment desired settings)
x=numpy.linspace(0, L, Nx)
y=numpy.linspace(0, W, Ny)
X, Y = numpy.meshgrid(x, y)

#       Steady state solver/plotter
#T,error=SteadySolve(T, (dx,dy), k, (conv,alpha), \
#                    (1,1,2,2), Tx1, Tx2, qy1, qy2)
#PlotXYT(X, Y, T, 300, 700)

#       Transient solver
for i in range(timeSteps):
    # Change a BC with time
#    if i%2==0:
#        Tx1=Tx1-50
#    T,error,Fo=TransSolve(1, T, (dx,dy), k, Fo, (conv,alpha),\
#                       (1,1,2,2), Tx1, Tx2, qy1, qy2)
    T,error,Fo=TransSolve(1, T, (dx,dy), k, Fo, (conv,alpha),\
                       (3,3,1,1), (hy1,Tinfy1), (hy1,Tinfy1), Tx1, Tx2)

    if error==1:
        print 'Convergence problem at time step %i'%i
        break

#    if i==timeSteps/2:
#        PlotXYT(X,Y,T,300,700)

PlotXYT(X, Y, T, 300, 700)

# 2D plots
#ploty=2
#pyplot.plot(X[1,:], T[int(ploty/dy),:])
#pyplot.xlabel('X')
#pyplot.ylabel('T')
##pyplot.title('Temperature distribution')
#
#plotx=2
#pyplot.figure()
#pyplot.plot(Y[:,1], T[:,int(plotx/dx)])
#pyplot.xlabel('Y')
#pyplot.ylabel('T')
##pyplot.title('Temperature distribution at X=%i'%plotx)