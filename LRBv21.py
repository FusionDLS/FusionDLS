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

from Iterate import LengFunc, iterate


class SimulationState():
    """
    This class represents the simulation state and contains all the variables and data 
    needed to run the simulation. The state is passed around different functions, which 
    allows more of the algorithm to be abstracted away from the main function.
    """
    def __init__(self):
        pass
    
    # Update many variables
    def update(self, **kwargs):
            self.__dict__.update(kwargs)

    # Will return this if called as string
    def __repr__(self):
        return str(self.__dict__)
    
class SimulationInputs():
    """
    This class functions the same as SimulationState, but is used to store the inputs instead.
    The separation is to make it easier to see which variables should be unchangeable.
    """
    def __init__(self):
        pass
    
    # Update many variables
    def update(self, **kwargs):
            self.__dict__.update(kwargs)

    # Will return this if called as string
    def __repr__(self):
        return str(self.__dict__)

def LRBv21(constants,radios,d,SparRange, 
                             control_variable = "impurity_frac",
                             verbosity = 0, Ctol = 1e-3, Ttol = 1e-2, 
                             URF = 1,
                             timeout = 20):
    """ function that returns the impurity fraction required for a given temperature at the target. Can request a low temperature at a given position to mimick a detachment front at that position.
    constants: dict of options
    radios: dict of options
    indexRange: array of S indices of the parallel front locations to solve for
    control_variable: either impurity_frac, density or power
    Ctol: error tolerance target for the inner loop (i.e. density/impurity/heat flux)
    Ttol: error tolerance target for the outer loop (i.e. rerrunning until Tu convergence)
    URF: under-relaxation factor for temperature. If URF is 0.2, Tu_new = Tu_old*0.8 + Tu_calculated*0.2. Always set to 1.
    Timeout: controls timeout for all three loops within the code. Each has different message on timeout. Default 20
    
    """
    # Initialise simulation state
    st = SimulationState()
    si = SimulationInputs()
    
    # Initialise state
    t0 = timer()
    splot = []
    st.error1 = 1
    st.error0 = 1
    output = defaultdict(list)

    # Lay out constants
    si.gamma_sheath = constants["gamma_sheath"]
    si.qpllu0 = constants["qpllu0"]
    si.nu0 = constants["nu0"]
    si.cz0 = constants["cz0"]
    si.Tt = constants["Tt"]
    si.Lfunc = constants["Lfunc"]
    si.alpha = constants["alpha"]
    si.verbosity = verbosity
    si.Ctol = Ctol
    si.Ttol = Ttol
    si.URF = URF
    si.timeout = timeout
    si.radios = radios
    si.control_variable = control_variable
    
    # Physics constants
    si.kappa0 = 2500 # Electron conductivity
    si.mi = 3*10**(-27) # Ion mass
    si.echarge = 1.60*10**(-19) # Electron charge
    
    # Extract topology data
    si.Xpoint = d["Xpoint"]
    si.S = d["S"]
    si.Spol = d["Spol"]
    si.Btot = d["Btot"]
    si.B = interpolate.interp1d(si.S, si.Btot, kind = "cubic")
    si.indexRange = [np.argmin(abs(d["S"] - x)) for x in SparRange] # Indices of topology arrays to solve code at

    # Initialise arrays for storing cooling curve data
    Tcool = np.linspace(0.3,500,1000)
    Lalpha = []
    for dT in Tcool:
        Lalpha.append(si.Lfunc(dT))
    Lalpha = np.array(Lalpha)
    Tcool = np.append(0,Tcool)
    Lalpha = np.append(0,Lalpha)
    si.Lz = [Tcool,Lalpha]
    
    # Calculation of radial heat transfer needed to achieve correct qpllu0 at Xpoint
    st.qradial = si.qpllu0/np.abs(si.S[-1]-si.S[si.Xpoint])

    print("Solving...", end = "")
    
    
    """------SOLVE------"""

    for point in si.indexRange: # For each detachment front location:
            
        print("{}...".format(point), end="")    
        
        if si.verbosity > 0:
            print("\n---SOLVING FOR INDEX {}".format(point))
            
        """------INITIAL GUESSES------"""
        
        # Current set of parallel position coordinates
        st.s = si.S[point:]
        output["Splot"].append(si.S[point])
        output["SpolPlot"].append(si.Spol[point])

        # Inital guess for the value of qpll integrated across connection length
        qavLguess = 0
        if si.radios["upstreamGrid"]:
            if st.s[0] < si.S[si.Xpoint]:
                qavLguess = ((si.qpllu0)*(si.S[si.Xpoint]-st.s[0]) + (si.qpllu0/2)*(st.s[-1]-si.S[si.Xpoint]))/(st.s[-1]-si.S[0])
            else:
                qavLguess = (si.qpllu0/2)
        else:
            qavLguess = (si.qpllu0)

        # Inital guess for upstream temperature based on guess of qpll ds integral
        Tu0 = ((7/2)*qavLguess*(st.s[-1]-st.s[0])/si.kappa0)**(2/7)
        st.Tu = Tu0
                                    
        # Cooling curve integral
        Lint = cumtrapz(si.Lz[1]*np.sqrt(si.Lz[0]),si.Lz[0],initial = 0)
        integralinterp = interpolate.interp1d(si.Lz[0],Lint)

        # Guesses/initialisations for control variables assuming qpll0 everywhere and qpll=0 at target
        if si.control_variable == "impurity_frac":
            cz0_guess = (si.qpllu0**2 )/(2*si.kappa0*si.nu0**2*st.Tu**2*integralinterp(st.Tu))
            st.cvar = cz0_guess
        elif si.control_variable == "density":
            st.cvar = si.nu0       
        elif si.control_variable == "power":
            st.cvar = 1/st.qradial #qpllu0
            
        # Initial guess of qpllt, the virtual target temperature (typically 0). 
        st.qpllt = si.gamma_sheath/2*si.nu0*st.Tu*si.echarge*np.sqrt(2*si.Tt*si.echarge/si.mi)
        
        
        """------INITIALISATION------"""
        
        log = defaultdict(list)
        st.error1 = 1 # Inner loop error (error in qpllu based on provided cz/ne)
        st.error0 = 1 # Outer loop residual in upstream temperature
        log["error1"].append(st.error1)
        
        # Tu convergence loop
        for k0 in range(si.timeout):
            
            # Initialise
            st = iterate(si, st)
            if si.verbosity > 1:
                print("\ncvar: {:.3E}, error1: {:.3E}".format(st.cvar, st.error1))

            """------INITIAL SOLUTION BOUNDING------"""

            # Double or halve cvar until the error flips sign
            log["cvar"].append(st.cvar)
            log["error1"].append(st.error1)
            log["qpllu1"].append(st.qpllu1)
            
            for k1 in range(si.timeout*2):
                
                if st.error1 > 0:
                    st.cvar = st.cvar / 2
                        
                elif st.error1 < 0:
                    st.cvar = st.cvar * 2

                st = iterate(si, st)

                log["cvar"].append(st.cvar)
                log["error1"].append(st.error1)
                log["qpllu1"].append(st.qpllu1)

                if si.verbosity > 1:
                    print("cvar: {:.3E}, error1: {:.3E}".format(st.cvar, st.error1))
    
                if si.verbosity > 2:
                    print("Last error: {:.3E}, New error: {:.3E}".format(log["error1"][k1+1], log["error1"][k1+2]))

                if np.sign(log["error1"][k1+1]) != np.sign(log["error1"][k1+2]): # It's initialised with a 1 already, hence k1+1 and k1+2
                    break
                    
                if k1 == si.timeout - 1:
                    print("******INITIAL BOUNDING TIMEOUT! Failed. Set verbosity = 3 if you want to diagnose.*******")


            if st.cvar < 1e-6 and si.control_variable == "impurity_fraction":
                print("*****REQUIRED IMPURITY FRACTION IS NEAR ZERO*******")
                #sys.exit()
                
            # We have bounded the problem -  the last two iterations
            # are on either side of the solution
            lower_bound = min(log["cvar"][-1], log["cvar"][-2])
            upper_bound = max(log["cvar"][-1], log["cvar"][-2])


            """------INNER LOOP------"""

            for k2 in range(si.timeout):

                # New cvar guess is halfway between the upper and lower bound.
                st.cvar = lower_bound + (upper_bound-lower_bound)/2
                st = iterate(si, st)
                log["cvar"].append(st.cvar)
                log["error1"].append(st.error1)
                log["qpllu1"].append(st.qpllu1)

                # Narrow bounds based on the results.
                if st.error1 < 0:
                    lower_bound = st.cvar
                elif st.error1 > 0:
                    upper_bound = st.cvar

                if si.verbosity > 1:
                    print(">Bounds: {:.3E}-{:.3E}, cvar: {:.3E}, error1: {:.3E}".format(
                        lower_bound, upper_bound, st.cvar, st.error1))

                if abs(st.error1) < si.Ctol:
                    break

                if k2 == si.timeout - 1:
                    print("******INNER LOOP TIMEOUT!*******")
                    #sys.exit()
                    
            # Calculate the new Tu by mixing new value with old one by factor URF (Under-relaxation factor)
            st.Tu = (1-si.URF)*st.Tu + si.URF*st.Tucalc
            st.error0 = (st.Tu-st.Tucalc)/st.Tu
            
            if si.verbosity > 0 :
                print("-----------error0: {:.3E}, Tu: {:.2f}, Tucalc: {:.2f}".format(st.error0, st.Tu, st.Tucalc))
                
            
            log["Tu"].append(st.Tu)
            log["error0"].append(st.error0)
            
            # Not sure if this Q serves any function
            Q = []
            for Tf in st.T:
                try:
                    Q.append(si.Lfunc(Tf))
                except:
                    print(f"FAILED TO QUERY COOLING CURVE for a temperature of {Tf:.3E}!")
                    break
                
            if abs(st.error0) < si.Ttol:
                break

            if k0 == si.timeout - 1:
                print("******OUTER TIMEOUT! Loosen Ttol or reduce under-relaxation factor. Set verbosity = 2!*******")
                #sys.exit()

            
        """------COLLECT PROFILE DATA------"""
        
        if si.control_variable == "power":
            output["cvar"].append(1/st.cvar) # so that output is in Wm-2
        else:
            output["cvar"].append(st.cvar)
            
        output["Tprofiles"].append(st.T)
        output["Sprofiles"].append(st.s)
        output["Qprofiles"].append(st.q)
        
        Qrad = []
        for Tf in st.T:
            if si.control_variable == "impurity_frac":
                Qrad.append(((si.nu0**2*st.Tu**2)/Tf**2)*st.cvar*si.Lfunc(Tf))
            elif si.control_variable == "density":
                Qrad.append(((st.cvar**2*st.Tu**2)/Tf**2)*si.cz0*si.Lfunc(Tf))
            elif si.control_variable == "power":
                Qrad.append(((si.nu0**2*st.Tu**2)/Tf**2)*si.cz0*si.Lfunc(Tf))
            
        output["Rprofiles"].append(Qrad)
        output["logs"].append(log)
        
    """------COLLECT RESULTS------"""
    # Here we calculate things like window, threshold etc from a whole scan.
    
    # Relative control variable:
    cvar_list = np.array(output["cvar"])
    crel_list = cvar_list / cvar_list[0]
    
    # S parallel and poloidal locations of each front location (for plotting against cvar/crel):
    splot = output["Splot"]
    spolplot = output["SpolPlot"]
    
    # Trim any unstable detachment (negative gradient) region for post-processing reasons 
    crel_list_trim = crel_list.copy()
    cvar_list_trim = cvar_list.copy()

    # Find values on either side of C = 1 and interpolate onto 1 
    for i in range(len(crel_list)-1):
        if np.sign(crel_list[i]-1) != np.sign(crel_list[i+1]-1) and i > 0:

            interp_par = interpolate.interp1d([crel_list[i], crel_list[i+1]], [splot[i], splot[i+1]])
            interp_pol = interpolate.interp1d([crel_list[i], crel_list[i+1]], [spolplot[i], spolplot[i+1]])
            
            spar_onset = float(interp_par(1))
            spol_onset = float(interp_pol(1))
            break
        if i == len(crel_list)-2:
            spar_onset = 0
            spol_onset = 0

    if len(crel_list)>1:
        grad = np.gradient(crel_list)
        for i, val in enumerate(grad):
            if i > 0 and np.sign(grad[i]) != np.sign(grad[i-1]):
                crel_list_trim[:i] = np.nan
                cvar_list_trim[:i] = np.nan
                
    # Pack things into the output dictionary.
    output["splot"] = splot
    output["indexRange"] = si.indexRange    
    output["cvar"] = cvar_list
    output["crel"] = crel_list
    output["cvar_trim"] = cvar_list_trim
    output["crel_trim"] = crel_list_trim
    output["threshold"] = cvar_list[0]
    output["window"] = cvar_list[-1] - cvar_list[0]
    output["window_ratio"] = cvar_list[-1] / cvar_list[0]
    output["spar_onset"] = spar_onset
    output["spol_onset"] = spol_onset
    output["constants"] = constants
    output["radios"] = si.radios
    
    # Convert back to regular dict
    output = dict(output)
    t1 = timer()
    
    print("Complete in {:.1f} seconds".format(t1-t0))
        
    return output
