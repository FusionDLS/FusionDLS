
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def refineGrid(p, Sfront, fine_ratio = 2, width = 2, resolution = None, diagnostic_plot = False):
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

    ## Create new cell widths and calculate new S array
    if  resolution == None:
        resolution = len(S)

    Snew = np.linspace(S[0], S[-1], resolution - 1)      # Estimate of new S. N-1 cause N(0) = 0
    dSnew = 1/(np.exp(-0.5 * ((Snew - Sfront)/(width))**2) * (fine_ratio-1) + 1)
    dSnew *= S[-1] / dSnew.sum()      # Normalise to the original S
    Snew = np.cumsum(np.insert(dSnew, 0, 0))
    
    ## Assemble new grid
    pnew = {}
    pnew["S"] = Snew
    for par in p.keys():
        if par != "Xpoint" and par != "S":
            pnew[par] = sp.interpolate.make_interp_spline(S, p[par], k = 2)

    pnew["Xpoint"] = np.argmin(np.abs(Snew - S[p["Xpoint"]]))
    
    
    ## Diagnostic plot
    if diagnostic_plot is True:
        fig, axes = plt.subplots(1,3, figsize = (15,5), dpi = 100)

        axes[0].plot(S, Snew, marker = "o", lw = 0)
        axes[0].plot(S, S, ls = "--", c = "k", lw = 1)
        axes[0].set_xlabel("Old Spar")
        axes[0].set_ylabel("New Spar")
        axes[0].set_title("Old vs. new Spar map")

        axes[1].plot(Snew[1:], dSnew)
        axes[1].set_xlabel("S")
        axes[1].set_ylabel("dS")
        axes[1].set_title("Grid width profile")

        axes[2].set_title("Btot profile")
        axes[2].plot(S, p["Btot"], marker = "o", lw = 0, ms = 6, markerfacecolor = "None", c = "darkorange", label = "Original")
        axes[2].plot(S, interpfun(S), c = "darkorange", lw = 1, label = "Original, interpolated")
        axes[2].plot(Snew, pnew["Btot"], marker = "o", ms = 3, lw = 0, label = "New grid")
        axes[2].scatter(S[p["Xpoint"]], p["Btot"][p["Xpoint"]], c = "r", label = "Old Xpoint", zorder = 100, s = 150, linewidths=1, marker = "x")
        axes[2].scatter(Snew[pnew["Xpoint"]], pnew["Btot"][pnew["Xpoint"]], c = "blue", label = "New Xpoint", zorder = 100, s = 250, linewidths=1, marker = "+")
        axes[2].set_xlabel("S [m]")
        axes[2].set_ylabel("Btot [T]")
        
        axes[2].legend(fontsize = 12)
        fig.tight_layout()
        
    return pnew