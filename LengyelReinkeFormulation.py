# %%
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad,trapz, cumtrapz, odeint, solve_ivp
from scipy import interpolate
import ThermalFrontFormulation as TF
from unpackConfigurations import unpackConfiguration,returnzl,returnll
from matplotlib.collections import LineCollection
import multiprocessing as mp


#Function to integrate, that returns dq/ds and dt/ds using Lengyel formulation and field line conduction
def LengFunc(y,s,kappa0,nu,Tu,cz,qpllu0,alpha,radios,S,B,Xpoint,Lfunc):
    qoverB,T = y
    #set density using constant pressure assumption
    ne = nu*Tu/T

    fieldValue = 0
    if s > S[-1]:
        fieldValue = B(S[-1])
    elif s< S[0]:
        fieldValue = B(S[0])
    else:
        fieldValue = B(s)
    #add a constant radial source of heat above the X point
    if radios["upstreamGrid"]:
        if s >S[Xpoint]:
            dqoverBds = ((nu**2*Tu**2)/T**2)*cz*Lfunc(T) - qpllu0/np.abs(S[-1]-S[Xpoint])
        else:
            dqoverBds = ((nu**2*Tu**2)/T**2)*cz*Lfunc(T) 
    else:
        dqoverBds = ((nu**2*Tu**2)/T**2)*cz*Lfunc(T) 
    # working on neutral/ionisation model
    dqoverBds = dqoverBds/fieldValue
    dtds = 0
    if radios["fluxlim"]:
        dtds = qoverB*fieldValue/(kappa0*T**(5/2)-qoverB*fieldValue*kappa0*T**(1/2)/(alpha*ne*np.sqrt(9E-31)))
    else:
        dtds = qoverB*fieldValue/(kappa0*T**(5/2))
    #return gradient of q and T
    return [dqoverBds,dtds]


def returnImpurityFracLeng(constants,radios,S,indexRange,dispBassum = False,dispqassum = False,dispUassum = False):
    """ function that returns the impurity fraction required for a given temperature at the target. Can request a low temperature at a given position to mimick a detachment front at that position."""
    C = []
    radfraction = []
    
    Tprofiles = []
    Sprofiles = []
    Qprofiles = []
    splot = []

    #lay out constants
    gamma_sheath = constants["gamma_sheath"]
    qpllu0 = constants["qpllu0"]
    nu = constants["nu"]
    kappa0 = constants["kappa0"]
    mi = constants["mi"]
    echarge = constants["echarge"]
    Tt = constants["Tt"]
    Xpoint = constants["XpointIndex"]
    B = constants["B"]
    Lfunc = constants["Lfunc"]
    alpha = constants["alpha"]

    #initialise arrays for storing cooling curve data
    Tcool = np.linspace(0.3,500,1000)#should be this for Ar? Ryoko 20201209 --> almost no effect
    Lalpha = []
    for dT in Tcool:
        Lalpha.append(Lfunc(dT))
    Lalpha = np.array(Lalpha)
    Tcool = np.append(0,Tcool)
    Lalpha = np.append(0,Lalpha)
    Lz = [Tcool,Lalpha]


    for i in indexRange:

        #current set of parallel position coordinates
        s = S[i:]
        splot.append(S[i])
        error0 = 1


        #inital guess for the value of qpll integrated across connection length
        qavLguess = 0
        if radios["upstreamGrid"]:
            if s[0] < S[Xpoint]:
                qavLguess = ((qpllu0)*(S[Xpoint]-s[0]) + (qpllu0/2)*(s[-1]-S[Xpoint]))/(s[-1]-S[0])
            else:
                qavLguess = (qpllu0/2)
        else:
            qavLguess = (qpllu0)

        #inital guess for upstream temperature based on guess of qpll ds integral
        Tu0 = ((7/2)*qavLguess*(s[-1]-s[0])/kappa0)**(2/7)
        Tu = Tu0

        #iterate through temperature until the consistent Tu is determined
        while np.abs(error0) > 0.001:


            Lint = cumtrapz(Lz[1]*np.sqrt(Lz[0]),Lz[0],initial = 0)
            integralinterp = interpolate.interp1d(Lz[0],Lint)
            #initial guess of cz0 assuming qpll0 everywhere and qpll=0 at target
            cz0 = (qpllu0**2 )/(2*kappa0*nu**2*Tu**2*integralinterp(Tu))
            cz = cz0
            #set initial percentage change of cz after every guess
            perChange = 0.5
            error1 = 1
            switch0 = 0
            swtich1 = 0
            swapped = 0
            #iterate until the correct impurity fraction is found
            while perChange >0.001:

                
                #initial guess of qpllt, typically 0
                qpllt = gamma_sheath/2*nu*Tu*echarge*np.sqrt(2*Tt*echarge/mi)
                
                result = odeint(LengFunc,y0=[qpllt/B(s[0]),Tt],t=s,args=(kappa0,nu,Tu,cz,qpllu0,alpha,radios,S,B,Xpoint,Lfunc))
                q = result[:,0]*B(s)
                T = result[:,1]
                # V = result[:,2]
                qpllu1 = q[-1]

                if radios["upstreamGrid"]:
                    error1 = (qpllu1-0)/qpllu0
                else:
                    error1 = (qpllu1-qpllu0)/qpllu0
                if 0<error1:
                    switch1 = 0
                    #increase counter 'swapped' if the guess overshoots
                    if switch1!=switch0:
                        swapped += 1
                    else:
                        swapped = 0
                    cz = cz*(1-perChange)
                else:
                    switch1 = 1
                    if switch1!=switch0:
                        swapped += 1
                    else:
                        swapped = 0
                    cz = cz*(1+perChange)
                switch0 = switch1
                #if the guess has overshot twice, decrease the change in cz per guess
                if swapped >1:
                    perChange = perChange*0.1
                    swapped = 0
            Tucalc = T[-1]
            Tu = 0.8*Tu + 0.2*Tucalc
            error0 = (Tu-Tucalc)/Tu


        Q = []
        for Tf in T:
            Q.append(Lfunc(Tf))

        C.append(np.sqrt(cz))
        Tprofiles.append(T)
        Sprofiles.append(s)
        Qprofiles.append(q)
    return splot, C, Sprofiles,Tprofiles,Qprofiles

