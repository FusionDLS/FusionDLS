from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad,trapz, cumtrapz, odeint, solve_ivp
from scipy import interpolate
from unpackConfigurationsMK import *
from matplotlib.collections import LineCollection
import multiprocessing as mp
from collections import defaultdict
from timeit import default_timer as timer
import pandas as pd
import sys

#Function to integrate, that returns dq/ds and dt/ds using Lengyel formulation and field line conduction
# def LengFunc(y,s,kappa0,nu,Tu,cz,qpllu0,alpha,radios,S,B,Xpoint,Lfunc,qradial):
def LengFunc(y, s, si, st):
    """
    Lengyel function. 
    This is passed to ODEINT in integrate() and used to solve for q and T along the field line.
    
    Inputs
    -------
    y : list
        List containing ratio of target q to target total B and target temperature 
    s : float
        Parallel coordinate of front position
    st : SimulationState
        Simulation state object containing all evolved parameters
    si : SimulationInput
        Simulation input object containing all constant parameters
    
    Outputs
    -------
    [dqoverBds,dtds] : list
        Heat flux gradient dq/ds and temperature gradient dT/ds
    """
    
    nu, Tu, cz, qradial = st.nu, st.Tu, st.cz, st.qradial
    kappa0, qpllu0, alpha, radios, S, B, Xpoint, Lfunc = si.kappa0, si.qpllu0, si.alpha, si.radios, si.S, si.B, si.Xpoint, si.Lfunc

    qoverB,T = y
    # set density using constant pressure assumption (missing factor of 2 at target due to lack of Bohm condition)
    ne = nu*Tu/T
    
    fieldValue = 0
    if s > S[-1]:
        fieldValue = B(S[-1])
    elif s < S[0]:
        fieldValue = B(S[0])
    else:
        fieldValue = B(s)
        
    # add a constant radial source of heat above the X point, which is qradial = qpll at Xpoint/np.abs(S[-1]-S[Xpoint]
    # i.e. radial heat entering SOL evenly spread between midplane and xpoint needs to be sufficient to get the 
    # correct qpll at the xpoint.
    
    if radios["upstreamGrid"]:
        if s >S[Xpoint]:
            # The second term here converts the x point qpar to a radial heat source acting between midplane and the xpoint
            try:
                dqoverBds = ((nu**2*Tu**2)/T**2)*cz*Lfunc(T) - qradial * fieldValue / B(S[Xpoint]) # account for flux expansion to Xpoint
            except:
                print("Failed. s: {:.2f}".format(s))
            else:
                dqoverBds = ((nu**2*Tu**2)/T**2)*cz*Lfunc(T) - qradial  
        else:
            dqoverBds = ((nu**2*Tu**2)/T**2)*cz*Lfunc(T) 
    else:
        dqoverBds = ((nu**2*Tu**2)/T**2)*cz*Lfunc(T) 
    
    # working on neutral/ionisation model
    dqoverBds = dqoverBds/fieldValue
    
    # Flux limiter
    dtds = 0
    if radios["fluxlim"]:
        dtds = qoverB*fieldValue/(kappa0*T**(5/2)-qoverB*fieldValue*kappa0*T**(1/2)/(alpha*ne*np.sqrt(9E-31)))
    else:
        dtds = qoverB*fieldValue/(kappa0*T**(5/2))
    #return gradient of q and T
    return [dqoverBds,dtds]

"""------ITERATOR FUNCTION------"""
# Define iterator function. This just solves the Lengyel function and unpacks the results.
# It must be defined here and not outside of the main function because it depends on many global
# variables.
    
def iterate(si, st):
    """
    Solves the Lengyel function for q and T profiles along field line.
    Calculates error1 by looking at upstream q and comparing it to 0 
    (when upstreamGrid=True) or to qpllu0 (when upstreamGrid=False).
    
    Inputs
    -------
    st : SimulationState
        Simulation state object containing all evolved parameters
    si : SimulationInput
        Simulation input object containing all constant parameters
    
    State modifications
    -------
    st.q : np.array
        Profile of heat flux along field line
    st.T : np.array
        Profile of temperature along field line
    st.Tucalc : float
        Upstream temperature for later use in outer loop to calculate error0
    st.qpllu1 : float
        Upstream heat flux
    st.error1 : float
        Error in upstream heat flux
        
    """
    if si.control_variable == "impurity_frac":
        st.cz = st.cvar
        st.nu = si.nu0

    elif si.control_variable == "density":
        st.cz = si.cz0
        st.nu = st.cvar

    si.Btot = [si.B(x) for x in si.S]
    st.qradial = si.qpllu0/ np.trapz(si.Btot[si.Xpoint:] / si.Btot[si.Xpoint], x = si.S[si.Xpoint:])
        
    if si.control_variable == "power":
        st.cz = si.cz0
        st.nu = si.nu0
        st.qradial = 1/st.cvar # This is needed so that too high a cvar gives positive error            

    if si.verbosity>2:
        print(f"qpllu0: {si.qpllu0:.3E} | nu: {st.nu:.3E} | Tu: {st.Tu:.1f} | cz: {st.cz:.3E} | cvar: {st.cvar:.2E}", end = "")

    result = odeint(LengFunc, 
                    y0 = [st.qpllt/si.B(st.s[0]),si.Tt],
                    t = st.s,
                    args = (si, st)
                    )
    out = dict()
    # Result returns integrals of [dqoverBds, dtds]
    st.q = result[:,0]*si.B(st.s)     # q profile
    st.T = result[:,1]                # Temp profile
    st.Tucalc = st.T[-1]              # Upstream temperature. becomes st.Tu in outer loop

    # Sometimes we get some negative q but ODEINT breaks down and makes upstream end positive.
    # If there are any negative values in the array, set the upstream q to the lowest value of q in array.
    # The algorithm will then know that it needs to go the other way
    if len(st.q[st.q<0]) > 0:
        st.qpllu1 = np.min(st.q) # minimum q
    else:
        st.qpllu1 = st.q[-1] # upstream q

    # If upstream grid, qpllu1 is at the midplane and is solved until it's 0. It then gets radial transport
    # so that the xpoint Q is qpllu0. If uypstramGrid=False, qpllu1 is solved to match qpllu0 at the Xpoint.
    if si.radios["upstreamGrid"]:
        st.error1 = (st.qpllu1 - 0)/si.qpllu0 
    else:
        st.error1 = (st.qpllu1 - si.qpllu0)/si.qpllu0

    if si.verbosity > 2:
        print(f" -> qpllu1: {st.qpllu1:.3E} | Tucalc: {st.Tucalc:.1f} | error1: {st.error1:.3E}")

    return st