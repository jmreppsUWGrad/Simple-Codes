# -*- coding: utf-8 -*-
"""
Started on Tue Sep 04 20:21:18 2018

@author: Joseph

This script contains transient solvers for 1D conduction:

The intention is to perform time-varying BC simulations.

REVISION Sept 20:
    -restructure solvers like that of 2D Conduction and 2D N-S solvers

FUNCTION INPUTS:
    BC_setup: 2 number array to designate BCs at x1, x2 boundaries respectively
            1-temp, 2-flux, 3-convective 
            e.g. 12-temp BCs at smallest x and flux at largest x
    bc1-array containing the BC info for smallest x
    bc2-array containing the BC info for largest x

"""
# Libraries
import numpy
from matplotlib import pyplot

# Transient, explicit solver with temperature BCs
#INPUTS: Number of nodes, total length, density, specific heat, 
#thermal conductivity, Fourier number, Number of time steps,  
#initial temp, temperature BCs

def TransExpSolve(T,dx,(rho, Cp, k),Fo,timeSteps,BC_setup,bc1,bc2):
    
    if BC_setup[0]==1:
        T[0]=bc1
    if BC_setup[1]==1:
        T[-1]=bc2
    if BC_setup[0]==3:
        Bi1=bc1[0]*dx/k
        Fo=bc1[2]/(Bi1+1)
    if BC_setup[1]==3:
        Bi2=bc2[0]*dx/k
        Fo=bc2[2]/(Bi2+1)
    dt=Fo*Cp*rho*dx**2/k
#    print 'Transient model (explicit) with Temperature BCs. dt= %.2fs for %i timesteps'% (dt,timeSteps)
#    print 'Temperature BCs of %.0f and %.0f'%(Tbc1, Tbc2)
    for i in range(timeSteps):
        Tc=T.copy()        
        # Calculate temperatures
        T[1:-1]=Fo*(Tc[:-2]+Tc[2:])+(1-2*Fo)*Tc[1:-1]
        
        if BC_setup[0]==2:
            T[0]=2*Fo*Tc[1]+Tc[0]*(1-2*Fo)+2*Fo*bc1*dx/k
        if BC_setup[1]==2:
            T[-1]=2*Fo*Tc[-2]+Tc[-1]*(1-2*Fo)+2*Fo*bc2*dx/k
        if BC_setup[0]==3:
            T[0]=2*Fo*Bi1*bc1[1]+Tc[0]*(1-2*Fo*Bi1-2*Fo)+2*Fo*Tc[1]
        if BC_setup[1]==3:
            T[-1]=2*Fo*Bi2*bc2[1]+Tc[-1]*(1-2*Fo*Bi1-2*Fo)+2*Fo*Tc[-2]
        #print(diff)
#    print '------------------------------------------------'
    return T

def TransImpSolve(T,dx,(rho, Cp, k),Fo,timeSteps,conv,BC_setup,bc1,bc2):
    
    if BC_setup[0]==1:
        T[0]=bc1
    if BC_setup[1]==1:
        T[-1]=bc2
    if BC_setup[0]==3:
        Bi1=bc1[0]*dx/k
    if BC_setup[1]==3:
        Bi2=bc2[0]*dx/k
    dt=Fo*Cp*rho*dx**2/k
    Tc=T.copy()
#    print 'Transient model (implicit) with Temperature BCs. dt=%.2fs for %i timesteps'%(dt,timeSteps)
#    print 'Temperature BCs of %.0f and %.0f'%(Tbc1, Tbc2)
    print 'Residuals:'
    for i in range(timeSteps):
        Tprev=T.copy()
        diff=10
        count=1
        print 'Time step %i:'%(i+1)
        # Calculate temperatures
        while (diff>conv) & (count<1000):
            Tc[1:-1]=(Fo*(T[:-2]+T[2:])+Tprev[1:-1])/(1+2*Fo)
            
            if BC_setup[0]==2:
                Tc[0]=(2*Fo*bc1*dx/k+2*Fo*Tc[1]+Tprev[0])/(1+2*Fo)
            if BC_setup[1]==2:
                Tc[-1]=(2*Fo*bc2*dx/k+2*Fo*Tc[-2]+Tprev[-1])/(1+2*Fo)
            if BC_setup[0]==3:
                Tc[0]=(2*Fo*Bi1*bc1[1]+2*Fo*Tc[1]+Tprev[0])/(1+2*Fo+2*Fo*Bi1)
            if BC_setup[0]==3:
                Tc[-1]=(2*Fo*Bi2*bc2[1]+2*Fo*Tc[-2]+Tprev[-1])/(1+2*Fo+2*Fo*Bi2)
                
            diff=numpy.sum(numpy.abs(T[:]-Tc[:]))/numpy.sum(numpy.abs(T[:]))
            count=count+1
            T=Tc.copy()
            print(diff)
        if count==1000:
            print 'Convergence problem'
            break
#    print '------------------------------------------------'
    return T


# ------------------Setup-----------------------------------
L=4.0 # Length of cylinder
k=25 # Thermal conductivity (W/m/K)
rho=8000 # Density (kg/m^3)
Cp=800 # Specific heat (J/kg/K)
N=5 # Number of nodes
Fo=0.01 # Fourier number (less than 0.5 for explicit method)

dx=L/(N-1)
T=numpy.zeros(N)

#       Convergence
conv=.001 # Convergence target (SS and implicit trans solvers)
Nt=2 # number of time steps (transient)
TranConvStab=0.5 # Fo(Bi+1)<=0.5 (trans, convective BC)
alpha=1 # Relaxation parameter (<1-stable, >1-faster)

#       Boundary and Initial conditions
T[1:-1]=300 # Initial temperature (uniform); transient and SS solvers
# Side 1 (heat flux, convection or temp)
qBC=1000 # Heat flux BC
T1=300 # Temp BC1
h=50 # Convective heat transfer coefficient (W/m^2/K)
Tinf=273 # Freestream temperature
# Side 2 (temp)
T2=300 # Temp BC2

# ----------------Solve and Plot
X=numpy.linspace(0,L,N)
count=1 # General use variable
#pyplot.plot(X,T)
for i in range(600, 900, 50):
#    T[0:2]=300
#    T=TransExpSolve(T,dx,(rho,Cp,k),Fo,Nt,(1,1),T1,i)
#    T=TransExpSolve(T,dx,(rho,Cp,k),Fo,timeSteps,(2,3),qBC,(h,Tinf,TranConvStab))
#    T=TransImpSolve(T,dx,(rho,Cp,k),Fo,Nt,conv,(1,1),T1,i)
#    T=TransImpSolve(T,dx,(rho,Cp,k),Fo,conv,timeSteps,(1,3),T1,(h,Tinf,TranConvStab))
    
    pyplot.plot(X, T, label='T2=%.0f'%i)
    pyplot.xlabel('X')
    pyplot.ylabel('T')
    pyplot.legend()
    count=count+1
pyplot.title('Temperature distribution, dt=%.2fs, %i time steps per BC'%(Fo*Cp*rho*dx**2/k,Nt))


