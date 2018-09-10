# -*- coding: utf-8 -*-
"""
Started on Tue Sep 04 20:21:18 2018

@author: Joseph

This script contains transient solvers for 1D conduction:

The intention is to perform time-varying BC simulations.

"""
# Libraries
import numpy
from matplotlib import pyplot

# Transient, explicit solver with temperature BCs
#INPUTS: Number of nodes, total length, density, specific heat, 
#thermal conductivity, Fourier number, Number of time steps,  
#initial temp, temperature BCs
def TransExpTempBCs(T, dx, rho, Cp, k, Fo, timeSteps, Tbc1, Tbc2):
    dt=Fo*Cp*rho*dx**2/k
    T[0]=Tbc1
    T[-1]=Tbc2
    print 'Transient model (explicit) with Temperature BCs. dt= %.2fs for %i timesteps'% (dt,timeSteps)
    print 'Temperature BCs of %.0f and %.0f'%(Tbc1, Tbc2)
    for i in range(timeSteps):
        Tc=T.copy()        
        # Calculate temperatures
        T[1:-1]=Fo*(Tc[:-2]+Tc[2:])+(1-2*Fo)*Tc[1:-1] 
        #print(diff)
    print '------------------------------------------------'
    return T

# Transient, implicit solver with temp BCs
def TransImpTempBCs(T, dx, rho, Cp, k, Fo, timeSteps, conv, Tbc1, Tbc2):
    dt=Fo*Cp*rho*dx**2/k
    T[0]=Tbc1
    T[-1]=Tbc2
    Tc=T.copy()
    print 'Transient model (implicit) with Temperature BCs. dt=%.2fs for %i timesteps'%(dt,timeSteps)
    print 'Temperature BCs of %.0f and %.0f'%(Tbc1, Tbc2)
    print 'Residuals:'
    for i in range(timeSteps):
        Tprev=T.copy()
        diff=10
        count=1
        print 'Time step %i:'%(i+1)
        # Calculate temperatures
        while (diff>conv) & (count<1000):
            Tc[1:-1]=(Fo*(T[:-2]+T[2:])+Tprev[1:-1])/(1+2*Fo)
            
            diff=numpy.sum(numpy.abs(T[:]-Tc[:]))/numpy.sum(numpy.abs(T[:]))
            count=count+1
            T=Tc.copy()
            print(diff)
        if count==1000:
            print 'Convergence problem'
            break
    print '------------------------------------------------'
    return T

# Transient, explicit solver with flux and temp BCs
#INPUTS: 

def TransExpFluxTempBCs(T, dx, rho, Cp, k, Fo, timeSteps, qbc1, Tbc2):
    dt=Fo*Cp*rho*dx**2/k
    T[-1]=Tbc2
    print 'Transient model (explicit) with Flux and Temperature BCs. dt=%.2fs for %i timesteps'%(dt,timeSteps)
    print 'Flux BC: %.0f and Temperature BC: %.0f'%(qbc1, Tbc2)
    for i in range(timeSteps):
        Tc=T.copy()        
        # Calculate temperatures
        T[1:-1]=Fo*(Tc[:-2]+Tc[2:])+(1-2*Fo)*Tc[1:-1] 
        T[0]=2*Fo*Tc[1]+Tc[0]*(1-2*Fo)+2*Fo*qbc1*dx/k
        #print(diff)
    print '------------------------------------------------'
    return T

# Transient, implicit solver with flux and temp BCs
def TransImpFluxTempBCs(T, dx, rho, Cp, k, Fo, timeSteps, conv, qbc1, Tbc2):
    dt=Fo*Cp*rho*dx**2/k
    T[-1]=Tbc2
    Tc=T.copy()
    print 'Transient model (implicit) with Flux and temperature BCs. dt=%.2fs for %i timesteps'%(dt,timeSteps)
    print 'Flux BC: %.0f and Temperature BC: %.0f'%(qbc1, Tbc2)
    print 'Residuals:'
    for i in range(timeSteps):
        Tprev=T.copy()
        diff=10
        count=1
        print 'Time step %i'%(i+1)
        # Calculate temperatures
        while (diff>conv) & (count<1000):
            Tc[1:-1]=(Fo*(T[:-2]+T[2:])+Tprev[1:-1])/(1+2*Fo)
            Tc[0]=(2*Fo*qbc1*dx/k+2*Fo*Tc[1]+Tprev[0])/(1+2*Fo)
            diff=numpy.sum(numpy.abs(T[:]-Tc[:]))/numpy.sum(numpy.abs(T[:]))
            count=count+1
            T=Tc.copy()
            print(diff)
        if count==1000:
            print 'Convergence problem'
            break
        #print(diff)
    print '------------------------------------------------'
    return T

# Transient, explicit solver with convective and temp BCs
#INPUTS:

def TransExpConvTempBCs(T, dx, rho, Cp, k, stab, timeSteps, h, Tinf, Tbc2):
    Bi=h*dx/k
    Fo=stab/(Bi+1)
    dt=Fo*Cp*rho*dx**2/k
    T[-1]=Tbc2
    print 'Transient model (explicit) with convective and Temperature BCs. dt=%.2fs for %i timesteps'%(dt,timeSteps)
    print 'Convective BC: h=%.1f, Tinf=%.0f and temperature BC: %.0f'%(h,Tinf,Tbc2)
    for i in range(timeSteps):
        Tc=T.copy()        
        # Calculate temperatures
        T[1:-1]=Fo*(Tc[:-2]+Tc[2:])+(1-2*Fo)*Tc[1:-1] 
        T[0]=2*Fo*Bi*Tinf+Tc[0]*(1-2*Fo*Bi-2*Fo)+2*Fo*Tc[1]
        #print(diff)
    print '------------------------------------------------'
    return T

# Transient, implicit solver with convective and temp BCs
def TransImpConvTempBCs(T, dx, rho, Cp, k, Fo, timeSteps, conv, h, Tinf, Tbc2):
    Bi=h*dx/k
    dt=Fo*Cp*rho*dx**2/k
    T[-1]=Tbc2
    Tc=T.copy()
    print 'Transient model (implicit) with convective and temperature BCs. dt=%.2fs for %i timesteps'%(dt,timeSteps)
    print 'Convective BC: h=%.1f, Tinf=%.0f and temperature BC: %.0f'%(h,Tinf,Tbc2)
    for i in range(timeSteps):
        Tprev=T.copy()
        diff=10
        count=1
        print 'Time step %i'%(i+1)
        # Calculate temperatures
        while (diff>conv) & (count<1000):
            Tc[1:-1]=(Fo*(T[:-2]+T[2:])+Tprev[1:-1])/(1+2*Fo)
            Tc[0]=(2*Fo*Bi*Tinf+2*Fo*Tc[1]+Tprev[0])/(1+2*Fo+2*Fo*Bi)
            diff=numpy.sum(numpy.abs(T[:]-Tc[:]))/numpy.sum(numpy.abs(T[:]))
            count=count+1
            T=Tc.copy()
            print(diff)
        if count==1000:
            print 'Convergence problem'
            break
        #print(diff)
    print '------------------------------------------------'
    return T
# ------------------Setup-----------------------------------
L=4.0 # Length of cylinder
k=25 # Thermal conductivity (W/m/K)
rho=8000 # Density (kg/m^3)
Cp=800 # Specific heat (J/kg/K)
N=5 # Number of nodes
Fo=0.2 # Fourier number (less than 0.5 for explicit method)

dx=L/(N-1)
T=numpy.zeros(N)

#       Convergence
conv=.001 # Convergence target (SS and implicit trans solvers)
timeSteps=5 # number of time steps (transient)
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
T2=600 # Temp BC2

# ----------------Solve and Plot
X=numpy.linspace(0,L,N)
count=1 # General use variable
#pyplot.plot(X,T)
for i in range(600, 900, 50):
#    T=TransExpTempBCs(T, dx, rho, Cp, k, Fo, timeSteps, T1, T2)
#    T=TransExpFluxTempBCs(T, dx, rho, Cp, k, Fo, timeSteps, qBC, T2)
#    T=TransExpConvTempBCs(T, dx, rho, Cp, k, TranConvStab, timeSteps, h, Tinf, T2)
    T=TransImpTempBCs(T, dx, rho, Cp, k, Fo, timeSteps, conv, T1, i)
#    T=TransImpFluxTempBCs(T, dx, rho, Cp, k, Fo, timeSteps, conv, qBC, T2)
#    T=TransImpConvTempBCs(T, dx, rho, Cp, k, Fo, timeSteps, conv, h, Tinf, T2)
    
    pyplot.plot(X, T, label='T2=%.0f'%i)
    pyplot.xlabel('X')
    pyplot.ylabel('T')
    pyplot.legend()
    count=count+1
pyplot.title('Temperature distribution, dt=%.2fs, %i time steps per BC'%(Fo*Cp*rho*dx**2/k,timeSteps))


