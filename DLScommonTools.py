from netCDF4 import Dataset
import pandas as pd
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from AnalyticCoolingCurves import *
from unpackConfigurationsMK import *
from matplotlib.collections import LineCollection
import os
import pickle as pkl
from LRBv21 import LRBv21
import matplotlib as mpl
import copy
import colorcet as cc
from scipy import interpolate
from matplotlib.ticker import FormatStrFormatter, StrMethodFormatter, MultipleLocator, FormatStrFormatter, AutoMinorLocator

# from unpackConfigurations import unpackConfiguration
# from LengyelReinkeFormulation import *
# import ThermalFrontFormulation as TF
# from LRBv2 import LRBv2

def scale_BxBt(Btot, Xpoint, scale_factor = 0, BxBt = 0):
# Scale a Btot profile to have an arbitrary flux expansion
# Specify either a scale factor or requried flux expansion
# TODO: MAKE SURE BPOL IS SCALED TOO

    Bt_base = Btot[0]
    Bx_base = Btot[Xpoint]
    BxBt_base = Bx_base / Bt_base
    
    if BxBt == 0 and scale_factor > 0:
        BxBt = BxBt_base * scale_factor
    elif BxBt == 0 and scale_factor == 0:
        print("Error - specify either scale factor or flux expansion")    

    # Keep Bx the same, scale Bt.
    # Calc new Bt based on desired BtBx
    Bt_new = 1/(BxBt / Bx_base)
    
    Btot_new = Btot * (Bx_base - Bt_new) / (Bx_base - Bt_base)
    
    # Translate to keep the same Bx as before
    transl_factor = Btot_new[Xpoint] - Bx_base
    Btot_new = Btot_new - transl_factor
    
    # Replace upstream of the Xpoint with the old data
    # So that we are only scaling downstream of Xpoint
    Btot_new[Xpoint:] = Btot[Xpoint:]
    
    return Btot_new

def scale_Lc(S_base, Spol_base, Xpoint, scale_factor=0, Lc = 0):
# Scale Spar and Spol profiles for arbitrary connection length
# Specify either a scale factor or requried connection length
# IMPLEMENT SPOL SCALING
    
    Lc_base = S_base[Xpoint]
    Lpol_base = Spol_base[Xpoint]
    
    # Having Lc non-zero 
    if Lc > 0 and scale_factor == 0:
        scale_factor = Lc / Lc_base
        # scale_factorPol = 
    elif Lc == 0 and scale_factor == 0:
        print("Error - specify either scale factor or connection length")

    # Scale up to get correct length
    S_new = S_base*scale_factor
    Spol_new = Spol_base*scale_factor

    # Align Xpoints
    S_new += S_base[Xpoint] - S_new[Xpoint]
    Spol_new += Spol_base[Xpoint] - Spol_new[Xpoint]

    # Make both the same upstream of Xpoint
    S_new[Xpoint:] = S_base[Xpoint:]
    Spol_new[Xpoint:] = Spol_base[Xpoint:]

    # Offset to make both targets at S = 0
    S_new -= S_new[0]
    Spol_new -= Spol_new[0]
    
    return S_new, Spol_new

def scale_Lm(S_base, Spol_base, Xpoint, scale_factor=0, Lm = 0):
# Scale Spar and Spol profiles above Xpoint for any midplane length
# Specify either a scale factor or requried connection length
# IMPLEMENT SPOL SCALING
    
    Lm_base = S_base[-1]
    Lpol_base = Spol_base[-1]
    
    # Having Lc non-zero 
    if Lm > 0 and scale_factor == 0:
        scale_factor = Lm / Lm_base
        # scale_factorPol = 
    elif Lm == 0 and scale_factor == 0:
        print("Error - specify either scale factor or connection length")

    # Scale up to get correct length
    S_new = S_base*scale_factor
    Spol_new = Spol_base*scale_factor

    # Align Xpoints
    S_new += S_base[Xpoint] - S_new[Xpoint]
    Spol_new += Spol_base[Xpoint] - Spol_new[Xpoint]

    # Make both the same upstream of Xpoint
    S_new[:Xpoint] = S_base[:Xpoint]
    Spol_new[:Xpoint] = Spol_base[:Xpoint]

    # Offset to make both targets at S = 0
    S_new -= S_new[0]
    Spol_new -= Spol_new[0]
    
    return S_new, Spol_new

def make_arrays(scan2d, list_BxBt_scales, list_Lc_scales, new =True, cvar = "ne", cut = True):
# Calculate 2D arrays of detachment window and threshold improvement

    if new == True: # New format for 2D scans
        
        arr = dict()

        arr["window"] = np.zeros((len(list_BxBt_scales), len(list_Lc_scales)))
        arr["threshold"] = np.zeros((len(list_BxBt_scales), len(list_Lc_scales)))
        arr["window_ratio"] = np.zeros((len(list_BxBt_scales), len(list_Lc_scales)))
        arr["threshold_scale"] = np.zeros((len(list_BxBt_scales), len(list_Lc_scales)))

        for col, BxBt in enumerate(list_BxBt_scales):
            for row, Lc in enumerate(list_Lc_scales):
                
                arr["window_ratio"][row,col] = scan2d[row][col]["window_ratio"]
                
                if cut == True:
                    if cvar == "q":
                        if arr["window_ratio"][row,col] <= 1:
                            arr["threshold"][row,col] = scan2d[row][col]["threshold"]
                            arr["window"][row,col] = scan2d[row][col]["window"]
                        else:
                            arr["threshold"][row,col] = np.nan
                            arr["window"][row,col] = np.nan
                            arr["window_ratio"][row,col] = np.nan
                    else:
                        if arr["window_ratio"][row,col] >= 1:
                            arr["threshold"][row,col] = scan2d[row][col]["threshold"]
                            arr["window"][row,col] = scan2d[row][col]["window"]
                        else:
                            arr["threshold"][row,col] = np.nan
                            arr["window"][row,col] = np.nan
                            arr["window_ratio"][row,col] = np.nan
                else:
                    arr["threshold"][row,col] = scan2d[row][col]["threshold"]
                    arr["window"][row,col] = scan2d[row][col]["window"]

        index_1_BxBt = np.where(list_BxBt_scales == 1)
        index_1_Lc = np.where(list_Lc_scales == 1)
        window_norm = arr["window"][index_1_BxBt,index_1_Lc]
        window_ratio_norm = arr["window_ratio"][index_1_BxBt,index_1_Lc]
        threshold_norm = arr["threshold"][index_1_BxBt,index_1_Lc]

        arr["window_norm"] = (arr["window"] - window_norm) / abs(window_norm) 
        arr["window_ratio_norm"] = (arr["window_ratio"] - window_ratio_norm) / abs(window_ratio_norm)
        arr["threshold_norm"] = arr["threshold"] / threshold_norm
        arr["threshold_norm"] -= 1
        
        arr["window_base"] = window_norm
        arr["window_ratio_base"] = window_ratio_norm
        arr["threshold_base"] = threshold_norm
        
    else:
        
        arr_window = []
        arr_threshold = []
        arr_window_ratio = []
        arr = dict()

        for i, BxBt_scale in enumerate(list_BxBt_scales):
            arr_window.append(scan2d[i]["window"])
            arr_threshold.append(scan2d[i]["threshold"])
            arr_window_ratio.append(scan2d[i]["window_ratio"])
        
        arr["window"] = np.array(arr_window) 
        arr["threshold"] = np.array(arr_threshold)
        arr["window_ratio"] = np.array(arr_window_ratio)

        index_1_BxBt = np.where(list_BxBt_scales == 1)
        index_1_Lc = np.where(list_Lc_scales == 1)
        window_norm = arr["window"][index_1_BxBt,index_1_Lc]
        threshold_norm = arr["threshold"][index_1_BxBt,index_1_Lc]
        print("yep")
        arr["window_norm"] = (arr["window"] - window_norm) / abs(window_norm) -1
        arr["threshold_norm"] = arr["threshold"] / threshold_norm
        arr["threshold_norm"] -= 1
        
        

    return arr


def make_window_band(d, o, spol_middle, size = 0.05, q = False):
    """Make detachment window band with a middle at the provided SPol coordinate
    The default window size is 5%"""
    # o = copy.deepcopy(o)
    # d = copy.deepcopy(d)
    
    band = dict()
    if q == False:
        crel = np.array(o["crel"])
    else:
        crel = 1/np.array(o["crel"])
    splot = np.array(o["Splot"])
    spolplot = np.array(o["SpolPlot"])
    Btot = d["Btot"]
    Btot_grad = np.gradient(Btot)

    c_grid = np.linspace(crel[0], crel[-1], 1000)

    spar_from_crel = interpolate.UnivariateSpline(crel, splot, k= 5)
    spol_from_crel = interpolate.UnivariateSpline(crel, spolplot, k= 5)
    crel_from_spol = interpolate.UnivariateSpline(spolplot, crel, k= 5)

    c_middle = crel_from_spol(spol_middle)
    
    band["C"] = [None] * 3
    band["C"][0] = c_middle * (1-size)
    band["C"][1] = c_middle
    band["C"][2] = c_middle * (1+size)

    for param in ["Spar", "Spol", "index", "R", "Z", "Btot"]:
        band[param] = np.array([float]*3)

    for i in range(3):
        band["Spar"][i] = spar_from_crel(band["C"][i])    
        band["Spol"][i] = spol_from_crel(band["C"][i])
        band["index"][i] = np.argmin(np.abs(d["S"] - band["Spar"][i]))
        band["R"][i] = d["R"][band["index"][i]]
        band["Z"][i] = d["Z"][band["index"][i]]
        band["Btot"][i] = d["Btot"][band["index"][i]]
        
    band["width_pol"] = band["Spol"][2] - band["Spol"][0]
    band["width_par"] = band["Spar"][2] - band["Spar"][0]
    band["Btot_avg"] = np.mean(Btot[band["index"][0]:band["index"][2]])
    band["Btot_grad_avg"] = np.mean(Btot_grad[band["index"][0]:band["index"][2]])

    return band

def file_write(data, filename):
# Writes an object to a pickle file.
    with open(filename, "wb") as file:
    # Open file in write binary mode, dump result to file
        pkl.dump(data, file)
        
        
        
def file_read(filename):
# Reads a pickle file and returns it.
    with open(filename, "rb") as filename:
    # Open file in read binary mode, dump file to result.
        data = pkl.load(filename)
        
    return data

def make_colors(number, cmap):
    """make_colors(number of colours, matplotlib colormap function)"""
    colors = []
    idx = np.linspace(0,255,number)
    
    for i in range(number):
        colors.append(cmap(int(idx[i])))
                      
    return colors

def mike_cmap(number):
    colors = ["teal", "darkorange", "firebrick",  "limegreen", "magenta","cyan", "navy"]
    return colors[:number]

def set_matplotlib_defaults():
    fontsize = 14
    plt.rc('font', size=fontsize) #controls default text size
    plt.rc('axes', titlesize=fontsize) #fontsize of the title
    plt.rc('axes', labelsize=fontsize) #fontsize of the x and y labels
    plt.rc('xtick', labelsize=fontsize) #fontsize of the x tick labels
    plt.rc('ytick', labelsize=fontsize) #fontsize of the y tick labels
    plt.rc('legend', fontsize=fontsize) #fontsize of the legend
    plt.rc('lines', linewidth=3)
    plt.rc('figure', figsize=(8,6))
    plt.rc('axes', grid = True)
    
    