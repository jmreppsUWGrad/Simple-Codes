# -*- coding: utf-8 -*-
"""
Started on Tue Sep 04 20:21:18 2018

@author: Joseph

This script contains solvers for 1D conduction:

Steady or transient (explicit and implicit)
with temperatures, heat flux or convective BCs
Relaxation parameter also coded (unsure of functionality)

"""
# Libraries
import numpy
from matplotlib import pyplot

# BC function Goal: Handle all BCs (one side at a time)
#INPUT: BC info (flux/temp/convection), node on boundary, 
#interior node(s), BC type (0,1,2), model type (SS or trans)
def ApplyBC(bc1, bc2, bc3, bc4, T2, T3, type_BC, model_type):
    if type_BC==0: # Temp BC (bc1 and T1 only)
        T1=bc1
    elif type_BC==1: # flux BC (all inputs)
        if model_type==0: # Steady model
            T1=bc1*bc2/bc3+T2
        elif model_type==10: # transient-explicit
            T1=2*bc1*T3+2*bc1*bc2*bc3/bc4+(1-2*bc1)*T2
        else: #                 transient-implicit
            T1=(2*bc1*bc2*bc3/bc4+2*bc1*T3+T2)/(1+2*bc1)
    else: # convective BC (most inputs)
        if model_type==0: # Steady model
            T1=(bc1*bc2+T2)/(bc1+1)
        elif model_type==10: # transient-explicit
            T1=2*bc1*bc2*bc3+T2*(1-2*bc1*bc2-2*bc1)+2*bc1*T3
        else: # transient-implicit
            T1=(2*bc1*bc2*bc3+2*bc1*T3+T2)/(1+2*bc1*bc2+2*bc1)
    
    return T1

# Steady state solver (Temperature BCs)
#INPUTS: Number of nodes, convergence criteria,
# initial temperature, temperature BCs (2)
def SSTempBCs(N, conv, alpha, Tinit, Tbc1, Tbc2):
    count=1
    diff=10
    T=numpy.zeros(N)
    T[1:-1]=Tinit
    T[0]=Tbc1
    T[-1]=Tbc2
    Tint=T.copy()
    print 'Steady state model with Temperature BCs \nResiduals:'
    
    while (diff>conv) & (count<1000):
        Tc=T.copy()
        # Calculate temperatures
        Tint[1:-1]=(Tc[2:]+Tc[:-2])/2
        T[1:-1]=alpha*Tint[1:-1]+(1-alpha)*Tc[1:-1] # Apply relaxation
        # Check convergence
        diff=numpy.sum(numpy.abs(T[:]-Tc[:]))/numpy.sum(numpy.abs(Tc[:]))
        count=count+1 # (steady)
        print(diff)
    
    return T

# Steady state solver with flux on one side
#INPUTS: Number of nodes, total length, convergence criteria, 
# initial temp, flux BC, temperature BC
def SSFluxTempBCs(N, L, conv, alpha, Tinit, qbc1, Tbc2):
    count=1
    diff=10
    dx=L/(N-1)
    T=numpy.zeros(N)
    T[1:-1]=Tinit
    T[0]=T[1]+qbc1*dx/k
    T[-1]=Tbc2
    Tint=T.copy()
    print 'Steady state model with Flux and temperature BC \nResiduals:'
    while (diff>conv) & (count<1000):
        Tc=T.copy()        
        # Calculate temperatures
        Tint[1:-1]=(Tc[2:]+Tc[:-2])/2
        T[1:-1]=alpha*Tint[1:-1]+(1-alpha)*Tc[1:-1] # Apply relaxation
        T[0]=T[1]+qbc1*dx/k
        # Check convergence
        diff=numpy.sum(numpy.abs(T[:]-Tc[:]))/numpy.sum(numpy.abs(Tc[:]))
        count=count+1 # (steady)
        print(diff)
    
    return T

# Steady state solver with convective end
#INPUTS: Number of nodes, total length, thermal conductivity, 
#convergence criteria, initial temp, convection coeff, 
#freestream temp, temperature BC

def SSConvTempBCs(N, L, k, conv, alpha, Tinit, h, Tinf, Tbc2):
    count=1
    diff=10
    dx=L/(N-1)
    T=numpy.zeros(N)
    Bi=h*dx/k
    T[:-1]=Tinit
    T[-1]=Tbc2
    Tint=T.copy()
    print 'Steady state model with convective and temperature BCs \nResiduals:'
    while (diff>conv) & (count<1000):
        Tc=T.copy()        
        # Calculate temperatures
        Tint[1:-1]=(Tc[2:]+Tc[:-2])/2
        T[1:-1]=alpha*Tint[1:-1]+(1-alpha)*Tc[1:-1] # Apply relaxation
        T[0]=(T[1]+Bi*Tinf)/(Bi+1)
        # Check convergence
        diff=numpy.sum(numpy.abs(T[:]-Tc[:]))/numpy.sum(numpy.abs(Tc[:]))
        count=count+1 # (steady)
        print(diff)
    return T

# Transient, explicit solver with temperature BCs
#INPUTS: Number of nodes, total length, density, specific heat, 
#thermal conductivity, Fourier number, Number of time steps,  
#initial temp, temperature BCs
def TransExpTempBCs(N, L, rho, Cp, k, Fo, timeSteps, Tinit, Tbc1, Tbc2):
    dx=L/(N-1)
    dt=Fo*Cp*rho*dx**2/k
    T=numpy.zeros(N)
    T[1:-1]=Tinit
    T[0]=Tbc1
    T[-1]=Tbc2
    print 'Transient model (explicit) with Temperature BCs. dt= %.2fs for %i timesteps'% (dt,timeSteps)
    for i in range(timeSteps):
        Tc=T.copy()        
        # Calculate temperatures
        T[1:-1]=Fo*(Tc[:-2]+Tc[2:])+(1-2*Fo)*Tc[1:-1] 
        #print(diff)
    
    return T

# Transient, implicit solver with temp BCs
def TransImpTempBCs(N, L, rho, Cp, k, Fo, timeSteps, conv, alpha, Tinit, Tbc1, Tbc2):
    dx=L/(N-1)
    dt=Fo*Cp*rho*dx**2/k
    T=numpy.zeros(N)
    T[1:-1]=Tinit
    T[0]=Tbc1
    T[-1]=Tbc2
    Tc=T.copy()
    Tc2=T.copy()
    print 'Transient model (implicit) with Temperature BCs. dt=%.2fs for %i timesteps\nResiduals:'%(dt,timeSteps)
    for i in range(timeSteps):
        Tprev=T.copy()
        diff=10
        count=1  
        print 'Time step %i:'%(i+1)
        # Calculate temperatures
        while (diff>conv) & (count<1000):
            Tc[1:-1]=(Fo*(T[:-2]+T[2:])+Tprev[1:-1])/(1+2*Fo)
            Tc2[1:-1]=alpha*Tc[1:-1]+(1-alpha)*Tprev[1:-1] # Apply relaxation
            diff=numpy.sum(numpy.abs(T[:]-Tc2[:]))/numpy.sum(numpy.abs(T[:]))
            count=count+1
            T=Tc2.copy()
            print(diff)
        if count==1000:
            print 'Convergence problem'
            break
    
    return T

# Transient, explicit solver with flux and temp BCs
#INPUTS: 

def TransExpFluxTempBCs(N, L, rho, Cp, k, Fo, timeSteps, Tinit, qbc1, Tbc2):
    dx=L/(N-1)
    dt=Fo*Cp*rho*dx**2/k
    T=numpy.zeros(N)
    T_infMed=300
    eta=numpy.zeros(N)
    v_0,v_1,v,N=0,0,0,0
    BC_changed=False
    T[:-1]=Tinit
    T[-1]=Tbc2
    print 'Transient model (explicit) with Flux and Temperature BCs. dt=%.2fs for %i timesteps'%(dt,timeSteps)
    for i in range(timeSteps):
        Tc=T.copy()
        v_0=numpy.sum(eta*dx)
        deta=4.89*10**6*(1-eta)*numpy.exp(-70000/8.314/Tc)
        eta+=deta*dt
        # Calculate temperatures
        T[1:-1]=Fo*(Tc[:-2]+Tc[2:])+(1-2*Fo)*Tc[1:-1]+deta[1:-1]*dx/k*4700000
        T[0]=2*Fo*Tc[1]+Tc[0]*(1-2*Fo)+2*Fo*qbc1*dx/k+deta[0]*dx/k*4700000
        
        # Semi-infinte medium
        T_infMed=300+2*qbc1/k*numpy.sqrt(k/rho/Cp*(i+1)*dt/numpy.pi)
        eta[eta<10**(-5)]=0
        v_1=numpy.sum(eta*dx)
        
#        if T[0]>1.5*T_infMed and not BC_changed:
        if numpy.amax(eta)>=0.8 and not BC_changed:
            print 'Ignition t=%f ms'%((i+1)*dt*1000)
            qbc1=0
            BC_changed=True
        elif (v_1-v_0)/dt>0.001:
            v+=(v_1-v_0)/dt
            N+=1
        #print(diff)
    try:
        print 'Average wave speed: %f'%(v/N)
    except:
        print 'No wave speed'        
    return T,eta,T_infMed

# Transient, implicit solver with flux and temp BCs
def TransImpFluxTempBCs(N, L, rho, Cp, k, Fo, timeSteps, conv, alpha, Tinit, qbc1, Tbc2):
    dx=L/(N-1)
    dt=Fo*Cp*rho*dx**2/k
    T=numpy.zeros(N)
    T[:-1]=Tinit
    T[-1]=Tbc2
    Tc=T.copy()
    Tc2=T.copy()
    print 'Transient model (implicit) with Flux and temperature BCs. dt=%.2fs for %i timesteps\nResiduals:'%(dt,timeSteps)
    for i in range(timeSteps):
        Tprev=T.copy()
        diff=10
        count=1
        print 'Time step %i'%(i+1)
        # Calculate temperatures
        while (diff>conv) & (count<1000):
            Tc[1:-1]=(Fo*(T[:-2]+T[2:])+Tprev[1:-1])/(1+2*Fo)
            Tc2[1:-1]=alpha*Tc[1:-1]+(1-alpha)*Tprev[1:-1] # Apply relaxation
            Tc2[0]=(2*Fo*qbc1*dx/k+2*Fo*Tc2[1]+Tprev[0])/(1+2*Fo)
            diff=numpy.sum(numpy.abs(T[:]-Tc2[:]))/numpy.sum(numpy.abs(T[:]))
            count=count+1
            T=Tc2.copy()
            print(diff)
        if count==1000:
            print('Convergence problem')
            break
        #print(diff)
    
    return T

# Transient, explicit solver with convective and temp BCs
#INPUTS:

def TransExpConvTempBCs(N, L, rho, Cp, k, stab, timeSteps, Tinit, h, Tinf, Tbc2):
    dx=L/(N-1)
    T=numpy.zeros(N)
    Bi=h*dx/k
    Fo=stab/(Bi+1)
    dt=Fo*Cp*rho*dx**2/k
    T[:-1]=Tinit
    T[-1]=Tbc2
    print 'Transient model (explicit) with convective and Temperature BCs. dt=%.2fs for %i timesteps'%(dt,timeSteps)
    for i in range(timeSteps):
        Tc=T.copy()        
        # Calculate temperatures
        T[1:-1]=Fo*(Tc[:-2]+Tc[2:])+(1-2*Fo)*Tc[1:-1] 
        T[0]=2*Fo*Bi*Tinf+Tc[0]*(1-2*Fo*Bi-2*Fo)+2*Fo*Tc[1]
        #print(diff)
    
    return T

# Transient, implicit solver with convective and temp BCs
def TransImpConvTempBCs(N, L, rho, Cp, k, Fo, timeSteps, conv, alpha, Tinit, h, Tinf, Tbc2):
    dx=L/(N-1)
    T=numpy.zeros(N)
    Bi=h*dx/k
    dt=Fo*Cp*rho*dx**2/k
    T[:-1]=Tinit
    T[-1]=Tbc2
    Tc=T.copy()
    Tc2=T.copy()
    print 'Transient model (implicit) with convective and temperature BCs. dt=%.2fs for %i timesteps\nResiduals:'%(dt,timeSteps)
    for i in range(timeSteps):
        Tprev=T.copy()
        diff=10
        count=1 
        print 'Time step %i'%(i+1)
        # Calculate temperatures
        while (diff>conv) & (count<1000):
            Tc[1:-1]=(Fo*(T[:-2]+T[2:])+Tprev[1:-1])/(1+2*Fo)
            Tc2[1:-1]=alpha*Tc[1:-1]+(1-alpha)*Tprev[1:-1] # Apply relaxation
            Tc2[0]=(2*Fo*Bi*Tinf+2*Fo*Tc2[1]+Tprev[0])/(1+2*Fo+2*Fo*Bi)
            diff=numpy.sum(numpy.abs(T[:]-Tc2[:]))/numpy.sum(numpy.abs(T[:]))
            count=count+1
            T=Tc2.copy()
            print(diff)
        if count==1000:
            print 'Convergence problem'
            break
        #print(diff)
    
    return T
# ------------------Setup-----------------------------------
L=0.001 # Length of cylinder
k=65 # Thermal conductivity (W/m/K)
rho=1523 # Density (kg/m^3)
Cp=625 # Specific heat (J/kg/K)
N=1000 # Number of nodes
Fo=0.2 # Fourier number (less than 0.5 for explicit method)

#       Convergence
conv=.001 # Convergence target (SS and implicit trans solvers)
timeSteps=19000 # number of time steps (transient)
TranConvStab=0.5 # Fo(Bi+1)<=0.5 (trans, convective BC)
alpha=1 # Relaxation parameter (<1-under, >1-over)

#       Boundary and Initial conditions
T_init=300 # Initial temperature (uniform); transient and SS solvers
# Side 1 (heat flux, convection or temp)
qBC=200000000 # Heat flux BC
T1=300 # Temp BC1
h=50 # Convective heat transfer coefficient (W/m^2/K)
Tinf=273 # Freestream temperature
# Side 2 (temp)
T2=300 # Temp BC2

# ----------------Solve and Plot (uncomment desired solvers)
X=numpy.linspace(0,L,N)
#T=SSTempBCs(N, conv, alpha, T_init, T1, T2)
#T=SSFluxTempBCs(N, L, conv, alpha, T_init, qBC, T2)
#T=SSConvTempBCs(N, L, k, conv, alpha, T_init, h, Tinf, T2)
#pyplot.plot(X, T, marker='x')
#T=TransExpTempBCs(N, L, rho, Cp, k, Fo, timeSteps, T_init, T1, T2)
T,eta,T_inf=TransExpFluxTempBCs(N, L, rho, Cp, k, Fo, timeSteps, T_init, qBC, T2)
#T=TransExpConvTempBCs(N, L, rho, Cp, k, TranConvStab, timeSteps, T_init, h, Tinf, T2)
pyplot.plot(X, T)
pyplot.plot(X[0], T_inf, marker='x')
pyplot.xlabel('X')
pyplot.ylabel('T')
#T=TransImpTempBCs(N, L, rho, Cp, k, Fo, timeSteps, conv, alpha, T_init, T1, T2)
#T=TransImpFluxTempBCs(N, L, rho, Cp, k, Fo, timeSteps, conv, alpha, T_init, qBC, T2)
#T=TransImpConvTempBCs(N, L, rho, Cp, k, Fo, timeSteps, conv, alpha, T_init, h, Tinf, T2)
#pyplot.plot(X, T, marker='o')


