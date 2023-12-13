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
    def __init__(self, si):
        
        self.si = si   # Add input object to state
        
        # Initialise variables
        self.error1 = 1
        self.error0 = 1
        self.qpllu1 = 0
        self.lower_bound = 0
        self.upper_bound = 0
        
        # Initialise log: one per front position index
        self.log = {}
        for i in si.indexRange:
            self.log[i] = defaultdict(list)
    
    ## Update primary log
    def update_log(self):
        for param in ["error0", "error1", "cvar", "qpllu1", "Tu", "lower_bound", "upper_bound"]:
            self.log[self.point][param].append(self.get(param))
            
        l = self.log[self.point]
        
        
        if self.si.verbosity >= 2:
            
            if len(l["error0"]) == 1:   # Print header on first iteration
                print(f"\n\n Solving at index {self.point}")
                print("--------------------------------")
                    
            print("error0: {:.3E}, Tu: {:.2f}, error1: {:.3E}, cvar: {:.3E}, lower_bound: {:.3E}, upper_bound: {:.3E}".format(
                l["error0"][-1], l["Tu"][-1], l["error1"][-1], l["cvar"][-1], l["lower_bound"][-1], l["upper_bound"][-1]))
            
    # Update many variables
    def update(self, **kwargs):
            self.__dict__.update(kwargs)

    # Will return this if called as string
    def __repr__(self):
        return str(self.__dict__)
    
    # Return parameter from state
    def get(self, param):
        return self.__dict__[param]
    
class SimulationInputs():
    """
    This class functions the same as SimulationState, but is used to store the inputs instead.
    The separation is to make it easier to see which variables should be unchangeable.
    """
    def __init__(self):
        # Physics constants
        self.kappa0 = 2500 # Electron conductivity
        self.mi = 3*10**(-27) # Ion mass
        self.echarge = 1.60*10**(-19) # Electron charge
        
        
    
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
    # Start timer
    t0 = timer()
    
    # Initialise simulation inputs object
    si = SimulationInputs()
    
    # Add inputs to SimulationInputs
    si.update(**constants)
    si.verbosity = verbosity
    si.Ctol = Ctol
    si.Ttol = Ttol
    si.URF = URF
    si.timeout = timeout
    si.radios = radios
    si.control_variable = control_variable
    
    # Extract topology data
    si.Xpoint = d["Xpoint"]
    si.S = d["S"]
    si.Spol = d["Spol"]
    si.Btot = d["Btot"]
    si.B = interpolate.interp1d(si.S, si.Btot, kind = "cubic")
    si.SparRange = SparRange
    si.indexRange = [np.argmin(abs(d["S"] - x)) for x in SparRange] # Indices of topology arrays to solve code at
    si.indexRange = np.unique(si.indexRange)   # Drop duplicates
    
    # Initialise simulation state object
    st = SimulationState(si)

    # Initialise output dictionary
    output = defaultdict(list)
 
    # Initialise cooling curve
    # TODO: move into SimulationInputs once interface is refactored
    Tcool = np.linspace(0.3,500,1000)
    Lalpha = []
    for dT in Tcool:
        Lalpha.append(si.Lfunc(dT))
    Lalpha = np.array(Lalpha)
    Tcool = np.append(0,Tcool)
    Lalpha = np.append(0,Lalpha)
    si.Lz = [Tcool,Lalpha]   # Array of temperatures and corresponding cooling

    print("Solving...", end = "")
    
    
    """------SOLVE------"""

    for point in si.indexRange: # For each detachment front location:
        st.point = point
            
        print("{}...".format(point), end="")    
            
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
            # Initial guess of cz0 assuming qpll0 everywhere and qpll=0 at target
            cz0_guess = (si.qpllu0**2 )/(2*si.kappa0*si.nu0**2*st.Tu**2*integralinterp(st.Tu))
            st.cvar = cz0_guess
            
        elif si.control_variable == "density":
            # Initial guess of nu0 assuming qpll0 everywhere and qpll=0 at target
            nu0_guess = np.sqrt((si.qpllu0**2 ) /(2*si.kappa0*si.cz0 * st.Tu**2 *integralinterp(st.Tu)))
            st.cvar = nu0_guess     
            
        elif si.control_variable == "power":
            # nu0 and cz0 guesses are from Lengyel which depends on an estimate of Tu using qpllu0
            # This means we cannot make a more clever guess for qpllu0 based on cz0 or nu0
            qpllu0_guess = si.qpllu0
            qradial_guess = qpllu0_guess / np.trapz(si.Btot[si.Xpoint:] / si.Btot[si.Xpoint], x = si.S[si.Xpoint:])
            st.cvar = 1/qradial_guess 
            
        # Initial guess of qpllt, the virtual target temperature (typically 0). 
        st.qpllt = si.gamma_sheath/2*si.nu0*st.Tu*si.echarge*np.sqrt(2*si.Tt*si.echarge/si.mi)
        
        
        """------INITIALISATION------"""
        st.error1 = 1 # Inner loop error (error in qpllu based on provided cz/ne)
        st.error0 = 1 # Outer loop residual in upstream temperature
        st.update_log()
        
        # Tu convergence loop
        for k0 in range(si.timeout):
            
            # Initialise
            st = iterate(si, st)

            """------INITIAL SOLUTION BOUNDING------"""

            # Double or halve cvar until the error flips sign           
            for k1 in range(si.timeout*2):
                
                if st.error1 > 0:
                    st.cvar = st.cvar / 2
                elif st.error1 < 0:
                    st.cvar = st.cvar * 2

                st = iterate(si, st)

                if np.sign(st.log[point]["error1"][k1+1]) != np.sign(st.log[point]["error1"][k1+2]): # It's initialised with a 1 already, hence k1+1 and k1+2
                    break
                    
                if k1 == si.timeout - 1: raise Exception("Initial bounding failed")
                
            if st.cvar < 1e-6 and si.control_variable == "impurity_fraction": raise Exception("Required impurity fraction is tending to zero")

                
            # We have bounded the problem -  the last two iterations
            # are on either side of the solution
            st.lower_bound = min(st.log[point]["cvar"][-1], st.log[point]["cvar"][-2])
            st.upper_bound = max(st.log[point]["cvar"][-1], st.log[point]["cvar"][-2])


            """------INNER LOOP------"""

            for k2 in range(si.timeout):

                # New cvar guess is halfway between the upper and lower bound.
                st.cvar = st.lower_bound + (st.upper_bound-st.lower_bound)/2
                st = iterate(si, st)

                # Narrow bounds based on the results.
                if st.error1 < 0:
                    st.lower_bound = st.cvar
                elif st.error1 > 0:
                    st.upper_bound = st.cvar

                # Break on success
                if abs(st.error1) < si.Ctol:
                    break

                if k2 == si.timeout - 1: print("WARNING: Failed to converge control variable loop")
                    
            """------OUTER LOOP------"""
            # Upstream temperature error
            st.error0 = (st.Tu-st.Tucalc)/st.Tu 
            
            # Calculate new Tu, under-relax by URF
            st.Tu = (1-si.URF)*st.Tu + si.URF*st.Tucalc

            st.update_log()
                
            # Break on outer (temperature) loop success
            if abs(st.error0) < si.Ttol:
                break

            if k0 == si.timeout - 1: raise Exception("Failed to converge temperature loop")

            
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
        output["logs"].append(st.log)
        
    """------COLLECT RESULTS------"""
    if len(SparRange) > 1:
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
        if len(crel_list)>1:
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
                    
            output["spar_onset"] = spar_onset
            output["spol_onset"] = spol_onset

        
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

        output["constants"] = constants
        output["radios"] = si.radios
        output["state"] = st
    
    # Convert back to regular dict
    output = dict(output)
    t1 = timer()
    
    print("Complete in {:.1f} seconds".format(t1-t0))
        
    return output
