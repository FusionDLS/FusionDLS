
from netCDF4 import Dataset
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import os, sys
import pickle as pkl
import scipy as sp

def get_sensitivity(crel_trim, SpolPlot, fluctuation=1.1, location=0, verbose = False):
        """
        Get detachment sensitivity at a certain location
        Sensitivity defined the location of front after a given fluctuation
        as a fraction of the total poloidal leg length.
        
        Inputs
        ------
        crel_trim: 1D array
            Crel values of detachment front with unstable regions trimmed (from DLS)
        SpolPlot: 1D array
            Poloidal distance from the DLS result
        fluctuation: float
            Fluctuation to calculate sensitivity as fraction of distance to X-point
            Default: 1.1
        location: float
            Location to calculate sensitivity as fraction of distance to X-point
            Default: target (0)
        verbose: bool
            Print results
            
        Returns
        -------
        sensitivity: float
            Sensitivity: position of front as fraction of distance towards X-point
            
        """
        # Drop NaNs for points in unstable region
        xy = pd.DataFrame()
        xy["crel"] = crel_trim
        xy["spol"] = SpolPlot
        xy = xy.dropna()

        Spol_from_crel = sp.interpolate.InterpolatedUnivariateSpline(xy["crel"], xy["spol"])
        Crel_from_spol = sp.interpolate.InterpolatedUnivariateSpline(xy["spol"], xy["crel"])

        Spol_at_loc = xy["spol"].iloc[-1] * location
        Crel_at_loc = Crel_from_spol(Spol_at_loc)
        Spol_total = xy["spol"].iloc[-1]


        if (Crel_at_loc - xy["crel"].iloc[0]) < -1e-6:
            sensitivity = 1   # Front in unstable region
        else:
            sensitivity = Spol_from_crel(Crel_at_loc*fluctuation) / Spol_total

        if verbose is True:
            print(f"Spol at location: {Spol_at_loc:.3f}")
            print(f"Crel at location: {Crel_at_loc:.3f}")
            print(f"Sensitivity: {sensitivity:.3f}")
            
        return sensitivity
    
    
    
def get_band_widths(d, o, cvar, size = 0.05):
    
    """ 
    Calculate detachment band widths based on geometry d and results o
    Size is the fraction of the band width to be calculated
    WORKS IN POLOIDAL
    """
    band_widths = {}
    # Find first valid index
    # This trims the unstable region on the inner
    # trim_idx = 0
    # for i,_ in enumerate(o["cvar_trim"]):
    #     if pd.isnull(o["cvar_trim"][i]) and not pd.isnull(o["cvar_trim"][i+1]):
    #         trim_idx = i+1
        
    # Make band based on topology dictionary (d) and results dictionary (o)
    # and a desired S poloidal location of the band centre (spol_middle)
    # as well as band size as a fraction (default +/-5%)
    trim_idx = 0
    crel = np.array(o["crel"])[trim_idx:]
    splot = np.array(o["Splot"])[trim_idx:]
    spolplot = np.array(o["SpolPlot"])[trim_idx:]

    if cvar == "power":
        crel = 1/crel

    c_grid = np.linspace(crel[0], crel[-1], 1000)
    k = 5
    spar_interp = sp.interpolate.interp1d(crel, splot, kind="cubic", fill_value = "extrapolate")
    spol_interp = sp.interpolate.interp1d(crel, spolplot, kind="cubic", fill_value = "extrapolate")

    band_widths = []
    for i, c_start in enumerate(c_grid):
        c_middle = c_start/(1-size)
        c_end = c_middle*(1+size)

        spar_start = spar_interp(c_start)
        spar_middle = spar_interp(c_middle)
        spar_end = spar_interp(c_end)

        spol_start = spol_interp(c_start)
        spol_middle = spol_interp(c_middle)
        spol_end = spol_interp(c_end)

        # band_widths.append(s_end - s_start)
        # band_widths.append(interp(c_start/(1-size)*(1+size)) - interp(c_start))
        band_width = spol_end - spol_start
        # band_width = spar_end - spar_start

        if band_width <= 0:
            band_widths.append(np.nan)
        elif spol_end > spolplot[-1]:
            band_widths.append(np.nan)
        else:
            band_widths.append(band_width)
            
    x = spol_interp(c_grid)
            
    return x, band_widths



def plot_morph_results(profiles, 
                       store, 
                       colors = ["teal", "darkorange"], 
                       xlabel = "Profile change",
                       show_47 = False,
                       sens = True):

    fig, axes = plt.subplots(2,3, dpi = 110, figsize = (12,8))

    
    thresholds = np.array([store[x]["threshold"] for x in profiles])
    windows = np.array([store[x]["window_ratio"] for x in profiles]) 
    L = np.array([profiles[x].get_connection_length() for x in profiles])
    BxBt = np.array([profiles[x].get_total_flux_expansion() for x in profiles])
    frac_gradB = np.array([profiles[x].get_average_frac_gradB() for x in profiles])
    avgB_ratio = np.array([profiles[x].get_average_B_ratio() for x in profiles])

    if sens:
        target_sens = np.array([get_sensitivity(store[x]["crel_trim"], store[x]["SpolPlot"], fluctuation=1.05, location=0.0) for x in profiles])

    L_base = L[0]
    BxBt_base = BxBt[0]
    threshold_base = thresholds[0]
    window_base = windows[0]
    avgB_ratio_base = avgB_ratio[0]
        
    threshcalc = (BxBt/BxBt_base)**(-1) * (L/L_base)**(-2/7) * (avgB_ratio/avgB_ratio_base)**(2/7)
    threshcalc2 = (BxBt/BxBt_base)**(-1) * (L/L_base)**(-4/7) * (avgB_ratio/avgB_ratio_base)**(2/7)

    windowcalc = (BxBt/BxBt_base)**(1) * (L/L_base)**(2/7) * (avgB_ratio/avgB_ratio_base)**(-2/7)
    windowcalc2 = (BxBt/BxBt_base)**(1) * (L/L_base)**(4/7) * (avgB_ratio/avgB_ratio_base)**(-2/7)

    index = list(profiles.keys())
    index /= index[0]

    # index = np.linspace(0,1,5)+i
    axes[0,0].set_title("Connection length")
    axes[0,0].plot(index, L/L_base, marker = "o")

    axes[0,1].set_title(r"Total flux expansion $\frac{B_{X}}{B_{t}}$")
    axes[0,1].plot(index, BxBt/BxBt_base, marker = "o")

    axes[0,2].set_title(r"Average $(\frac{1}{B}) \frac{dB}{ds_{pol}}$ below X-point")
    axes[0,2].plot(index, frac_gradB, marker = "o")

    axes[1,0].set_title(r"Detachment threshold")
    axes[1,0].plot(index, thresholds/threshold_base, marker = "o")

    if show_47:
        label_27 = "Analytical, $L^{2/7}$"
    else:
        label_27 = "Analytical"
    axes[1,0].plot(index, threshcalc, color = "darkslategrey", ls = "--", label = label_27, zorder = 100)
    
    if show_47:
        axes[1,0].plot(index, threshcalc2, color = "darkslategrey", ls = ":", label = "Analytical, $L^{4/7}$", zorder = 100)
    axes[1,0].legend(fontsize = "x-small")

    axes[1,1].set_title(r"Detachment window")
    axes[1,1].plot(index, windows/window_base, marker = "o")
    axes[1,1].plot(index, windowcalc, color = "darkslategrey", ls = "--", label = label_27, zorder = 100)
    
    if show_47:
        axes[1,1].plot(index, windowcalc2, color = "darkslategrey", ls = ":", label = "Analytical, $L^{4/7}$", zorder = 100)
    axes[1,1].legend(fontsize = "x-small")


    axes[1,2].set_title(r"10% Sensitivity @ target")
    
    if sens: 
        axes[1,2].plot(index, target_sens, marker = "o")

    for ax in axes.flatten():
        ax.set_xlabel(xlabel)

    for ax in [axes[0,0], axes[0,1], axes[1,0], axes[1,1]]:
        ax.set_ylabel("Relative change")
        
    axes[0,2].set_ylabel(r"$(\frac{1}{B}) \frac{dB}{ds_{pol}}$")
    axes[1,2].set_ylabel("$S_{front, final} \ /\  S_{pol, x}$")
    fig.tight_layout()