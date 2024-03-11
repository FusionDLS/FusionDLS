import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from DLScommonTools import pad_profile

def refineGrid(p, 
               Sfront, 
               fine_ratio = 1.5, 
               width = 4, 
               resolution = None, 
               diagnostic_plot = False,
               tolerance = 1e-3,
               timeout = 50):
    """
    Refines the grid around the front location.
    Refinement is in the form of a Gaussian distribution
    with a peak determined by fine_ratio and a given width.
    
    Inputs
    -----
    p: dict
        Dictionary containing the profile data (S, Spol, Btot, Bpol, Xpoint, R, Z)
    fine_ratio: float, default 2
        Ratio of coarse cell size to fine cell size
    width: float, default 2m
        Width of the fine region in meters parallel   
    resolution: float, default None
        resolution of resulting grid. If None, use same resolution as original grid. 
    
    Returns
    -----
    pnew: dict
        New dictionary containing the same profile data as p
    """
    
    S = p["S"]

    if  resolution == None:
        resolution = len(S)

    ## Grid generation is an iterative process because dSnew must know where to put the gaussian
    # refinement in S space, so it needs an initial S guess. Then we calculate new S from the dS
    # and loop again until it stops changing.
    Snew = np.linspace(S[0], S[-1], resolution - 1)      # Initialise S with uniform spacing
    residual = 1
    
    if diagnostic_plot is True:
        fig, axes = plt.subplots(2,1, figsize = (5,5), height_ratios = (8,4))
        
    for i in range(timeout):
        dSnew = 1/((width*np.sqrt(2*np.pi)) * np.exp(-0.5 * ((Snew- Sfront)/(width))**2) * (fine_ratio-1) + 1)
        dSnew *= S[-1] / dSnew.sum()      # Normalise to the original S
        Snew = np.cumsum(dSnew)
        if i != 0:
            residual = abs((dSnew2[-1] - dSnew[-1]) / dSnew2[-1])
        dSnew2 = dSnew
        
        if diagnostic_plot is True:
            axes[0].plot(Snew, dSnew, label = i)
            axes[1].scatter(Snew, np.ones_like(Snew)*-i, marker = "|", s = 5, linewidths = 0.5, alpha = 0.1)
        
        if residual < tolerance:
            # print(f"Residual is {residual}, breaking")
            break
        
        if i == timeout-1:
            raise Exception("Iterative grid adaption iteration limit reached, try reducing refinement ratio and running with diagnostic plot")
    
    Snew = np.insert(Snew, 0, 0)   # len(dS) = len(S) - 1
    
    # Grid width diagnostics plot settings
    if diagnostic_plot is True:
        
        axes[1].set_yticklabels([])
        fig.tight_layout()
        fig.legend(loc = "upper center", bbox_to_anchor = (0.5, 0), ncols = 5)
        axes[0].set_title("Adaptive grid iterations")

        axes[0].set_ylabel("dS [m]")
        axes[0].set_xlabel("S [m]")
        axes[1].set_title("S spacing")
        axes[1].set_xlabel("S [m]")
        fig.tight_layout()
    
    ## Interpolate geometry and field onto the new S coordinate
    pnew = {}
    pnew["S"] = Snew
    for par in ["S", "Spol", "R", "Z", "Btot", "Bpol"]:
        if par != "Xpoint" and par != "S":
            pnew[par] = sp.interpolate.make_interp_spline(S, p[par], k = 2)(Snew)
            
            if diagnostic_plot is True:
                fig, ax = plt.subplots(dpi = 100)
                ax.plot(p["S"], p[par], label = "Original", marker = "o", 
                        color = "darkorange", alpha = 0.3, 
                        ms = 10, markerfacecolor = "None")
                ax.plot(pnew["S"], pnew[par], label = "New", marker = "o", ms = 3)
                ax.set_title(par)

    pnew["Xpoint"] = np.argmin(np.abs(Snew - S[p["Xpoint"]]))
    
    
        
    return pnew