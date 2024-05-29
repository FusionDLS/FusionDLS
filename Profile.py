import copy
from netCDF4 import Dataset
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import os, sys
import pickle as pkl
import scipy as sp


class Profile():
    """
    Class defining a single field line profile (field line with topology)
    """
    
    def __init__(self, R, Z, Xpoint, Btot, Bpol, S, Spol, name = "base"):
        self.R = R
        self.Z = Z
        self.Xpoint = Xpoint
        
        # Split into leg up to and incl. xpoint
        self.R_leg = R[:Xpoint+1]
        self.Z_leg = Z[:Xpoint+1]
        
        self.Btot = Btot
        self.Bpol = Bpol
        self.S = S
        self.Spol = Spol
        
        self.name = name
    
    ## Allows to get attributes, set attributes as if it was a dict
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)
        
    def keys(self):
        return self.__dict__.keys()
        
    ## Allows copying
    def copy(self):
        # obj = type(self).__new__(self.__class__)
        # obj.__dict__.update(self.__dict__)
        # return obj
    
        return copy.deepcopy(self)
        
        
    def get_connection_length(self):
        """ 
        Return connection length of profile
        """
        return self.S[-1] - self.S[0]
    
    
    def get_total_flux_expansion(self):
        """
        Return total flux expansion of profile
        """
        return self.Btot[self.Xpoint] / self.Btot[0]
    
    
    def get_average_frac_gradB(self):
        """
        Return the average fractional Btot gradient
        below the X-point
        """
        return ((np.gradient(self.Btot, self.Spol) / self.Btot)[:self.Xpoint]).mean()
    
    def get_gradB_integral(self):
        """
        Return the integral of the fractional Btot gradient
        below the X-point
        """
        return np.trapz((np.gradient(self.Btot, self.Spol) / self.Btot)[:self.Xpoint], self.Spol[:self.Xpoint])
    
    def get_gradB_average(self):
        """
        Return the integral of the fractional Btot gradient
        below the X-point
        """
        return np.mean((np.gradient(self.Btot, self.Spol) / self.Btot)[:self.Xpoint])
    
    def get_Bpitch_integral(self):
        """
        Return the integral of the pitch angle Bpol/Btot
        below the X-point
        """
        return np.trapz((self.Bpol/self.Btot)[:self.Xpoint], self.Spol[:self.Xpoint])
    
    def get_Bpitch_average(self):
        """
        Return the integral of the pitch angle Bpol/Btot
        below the X-point
        """
        return np.mean((self.Bpol/self.Btot)[:self.Xpoint])
    
    
    def get_average_B_ratio(self):
        """
        Return the average Btot below X-point
        """
        return self.Btot[self.Xpoint] / (self.Btot[:self.Xpoint]).mean()
    

    
    def scale_BxBt(self, scale_factor = None, BxBt = None, verbose = True):
        """
        Scale a Btot profile to have an arbitrary flux expansion
        Specify either a scale factor or requried flux expansion
        Will keep Spol the same (not pitch angle)
        Will not modify R,Z coordinates!
        """

        Btot = self.Btot
        breakpoint = self.Xpoint + 1
        
        Bt_base = Btot[0]
        Bx = Btot[self.Xpoint]
        BxBt_base = Bx / Bt_base
        
        if BxBt == None and scale_factor == None:
            raise ValueError("Specify either scale factor or flux expansion")   
        elif BxBt == None:
            BxBt = BxBt_base * scale_factor
         
        if BxBt == 0:
            raise ValueError("BxBt cannot be 0")

        # Keep Bx the same, scale Bt.
        # Calc new Bt based on desired BtBx
        Bt_new = 1/(BxBt / Bx)
        
        Btot_upstream = Btot[breakpoint:]
        Btot_leg = Btot[:breakpoint]
        
        old_span = Bx - Bt_base
        new_span = Bx - Bt_new
        Btot_leg_new = Btot_leg * (new_span / old_span)   # Scale to get Bx/Bt ratio
        Btot_leg_new = Btot_leg_new - (Btot_leg_new[-1] - Bx)  # Offset to realign Bx
        
        if verbose is True:
            print("Warning: scaling flux expansion. R,Z coordinates will no longer be physical")
        
        self.Btot = np.concatenate((Btot_leg_new, Btot_upstream))
        
        
    def scale_Lc(self, scale_factor = None, Lc = None, verbose = True):
        """
        Scale Spar and Spol profiles for arbitrary connection length
        Specify either a scale factor or requried connection length
        Will keep Spol the same (not pitch angle)
        Will not modify R,Z coordinates!
        """
        # FIXME looks to have about 5% error when using Lc
        
        if scale_factor == None and Lc == None:
            raise ValueError("Specify either scale factor or connection length")
        
        breakpoint = self.Xpoint+1

        def scale_leg(S, scale_factor):
            L_upstream = S[-1] - S[breakpoint]
            L_leg = S[breakpoint]
            L_total = S[-1]
            
            if Lc == None:
                L_leg_new = L_total * scale_factor - L_upstream
            else:
                L_leg_new = Lc - L_upstream

            S_leg = S[:breakpoint]
            S_upstream = S[breakpoint:]
            S_leg_new = S_leg * L_leg_new / L_leg
            S_upstream_new = S_upstream + S_leg_new[-1] - S_leg[-1]
            S_new = np.concatenate((S_leg_new, S_upstream_new))
            
            return S_new
            
        self["S"] = scale_leg(self["S"], scale_factor)
        self["Spol"] = scale_leg(self["Spol"], scale_factor)
        
        if verbose is True:
            print("Warning: Scaling connection length. R,Z coordinates will no longer be valid")
        
    
    
    def offset_control_points(self, offsets, factor = 1):
        """
        Take profile and add control points [x,y] 
        Then perform cord spline interpolation to get interpolated profile in [xs,ys]
        The degree of the morph can be controlled by the factor
        Saves control points as R_control, Z_control
        """

        self.R_control, self.Z_control = shift_points(self.R_leg, self.Z_leg, offsets, factor = factor)    # Control points defining profile
        
        self.interpolate_leg_from_control_points()
        self.recalculate_topology()
        
        
    def interpolate_leg_from_control_points(self):
        """
        Takes saved R_control and Z_control and uses them to interpolate new R,Z 
        coordinates for the entire profile
        Saves new R and Z as well as the leg interpolations R_leg_spline and Z_leg_spline
        """
        
        self.R_leg_spline, self.Z_leg_spline = cord_spline(self.R_control, self.Z_control)   # Interpolate
        
        ## Calculate the new leg RZ from the spline
        # Note xs and ys are backwards
        dist = get_cord_distance(self["R_leg"], self["Z_leg"])   # Distances along old leg
        spl = cord_spline(self["R_control"][::-1], self["Z_control"][::-1], return_spline = True)
            
        # spl = cord_spline(self["R_leg_spline"][::-1], self["Z_leg_spline"][::-1], return_spline = True)   # Spline interp for new leg
        self["R_leg_spline"], self["Z_leg_spline"] = spl(dist)     # New leg interpolated onto same points as old leg
    
    
        ## Calculate total RZ by adding upstream
        self["R"] = np.concatenate([
            self["R_leg_spline"],
            self["R"][self["Xpoint"]+1:], 
            ])
        
        self["Z"] = np.concatenate([
            self["Z_leg_spline"],
            self["Z"][self["Xpoint"]+1:], 
            ])
        
        
    def recalculate_topology(self, constant_pitch = True, Bpol_shift = None):
        """ 
        Recalculate Spol, S, Btor, Bpol and Btot from R,Z
        If doing this after morphing a profile:
        - It requires R_leg and Z_leg to be the original leg
        - The new leg is contained in R_leg_spline and Z_leg_spline
        - The above are used to calculate new topology 
        Currently only supports changing topology below the X-point
        
        Bpol_shift: dict()
            Width = gaussian width in m
            pos = position in m poloidal from the target
            height = height in Bpol units
        """

        ## Calculate existing toroidal field (1/R)
        Btor = np.sqrt(self["Btot"]**2 - self["Bpol"]**2)   # Toroidal field
        Bpitch = self["Bpol"] / self["Btot"]
        
        ## Save existing parameters
        self["Bpol"] = self["Bpol"].copy()  
        Btor_leg = Btor[:self["Xpoint"]+1]
        Bpol_leg = self["Bpol"][:self["Xpoint"]+1]   
        Bpitch_leg = Bpitch[:self["Xpoint"]+1]  
        
        ## Calculate new S poloidal from R,Z
        self["Spol"] = returnll(self["R"], self["Z"])
        
        ## Calculate toroidal field (1/R)
        Btor_leg_new = Btor_leg * (self["R_leg"] / self["R_leg_spline"])

        ## Calculate total field
        # Either keep same Bpitch or same Bpol
        if constant_pitch is True:
            Btot_leg_new = np.sqrt(Btor_leg_new**2 / (1 - Bpitch_leg**2))
            Bpol_leg_new = np.sqrt(Btot_leg_new**2 - Btor_leg_new**2)
            
            self["Bpol"] = np.concatenate([
                Bpol_leg_new,
                self["Bpol"][self["Xpoint"]+1:], 
                ])
            
            ## Convolve Bpol with a gaussian of a width, position and height
            if Bpol_shift != None:
                width = Bpol_shift["width"]
                pos = Bpol_shift["pos"]
                height = Bpol_shift["height"]
                weight = (width*np.sqrt(2*np.pi)) * np.exp(-0.5 * ((np.array(self["Spol"]) - pos)/(width))**2) 
                weight = weight / np.max(weight) * height
                self["Bpol"] -= weight
            
        else:
            Btot_leg_new = np.sqrt(Btor_leg_new**2 + Bpol_leg**2)
        
        self["Btot"] = np.concatenate([
            Btot_leg_new,
            self["Btot"][self["Xpoint"]+1:], 
            ])
        
        ## Calculate parallel connection length
        self["S"] = returnS(self["R"], self["Z"], self["Btot"], self["Bpol"])
        
        # Recalculate new R_leg and Z_leg. Must be at the end to preserve reference
        # to old leg for Btor
        self["R_leg"] = self["R"][:self["Xpoint"]+1]
        self["Z_leg"] = self["Z"][:self["Xpoint"]+1]
        
        
    def plot_topology(self):

        fig, axes = plt.subplots(2,2, figsize = (8,8))

        basestyle = dict(c = "black")
        xstyle = dict(marker = "+", linewidth = 2, s = 150, c = "r", zorder = 100)

        ax = axes[0,0]
        ax.set_title("Fractional $B_{tot}$ gradient")
        ax.plot(self["Spol"], np.gradient(self["Btot"], self["Spol"]) / self["Btot"], **basestyle)
        ax.scatter(self["Spol"][self["Xpoint"]], (np.gradient(self["Btot"], self["Spol"]) / self["Btot"])[self["Xpoint"]], **xstyle)
        ax.set_xlabel(r"$S_{\theta} \   [m]$");   
        ax.set_ylabel("$B_{tot}$ $[T]$")


        ax = axes[1,0]
        ax.set_title("$B_{tot}$")
        ax.plot(self["Spol"], self["Btot"], **basestyle)
        ax.scatter(self["Spol"][self["Xpoint"]], self["Btot"][self["Xpoint"]], **xstyle)
        ax.set_xlabel(r"$S_{\theta} \   [m]$")
        ax.set_ylabel("$B_{tot}$ $[T]$")


        ax = axes[0,1]
        ax.set_title(r"Field line pitch $B_{pol}/B_{tot}$")
        ax.plot(self["Spol"], self["Bpol"]/self["Btot"], **basestyle)
        ax.scatter(self["Spol"][self["Xpoint"]], (self["Bpol"]/self["Btot"])[self["Xpoint"]], **xstyle)
        ax.set_xlabel(r"$S_{\theta} \   [m]$")
        ax.set_ylabel(r"$B_{pol} \ / B_{tot}$ ")

        ax = axes[1,1]
        ax.set_title("$B_{pol}$")
        ax.plot(self["Spol"], self["Bpol"], **basestyle)
        ax.scatter(self["Spol"][self["Xpoint"]],  (self["Bpol"])[self["Xpoint"]], **xstyle)
        ax.set_xlabel(r"$S_{\theta} \   [m]$")
        ax.set_ylabel(r"$B_{\theta}$ $[T]$")

        fig.tight_layout()
        
        
    def plot_control_points(
        self, 
        linesettings = {}, 
        markersettings = {}, 
        ylim = (None, None), 
        xlim = (None, None), 
        dpi = 100,
        ax = None,
        ):
        
        if ax == None:
            fig, ax = plt.subplots(dpi = dpi)
            ax.plot(self["R"], self["Z"], linewidth = 3, marker = "o", markersize = 0, color = "black", alpha = 1)
        
        default_line_args = {"c" : "forestgreen", "alpha" : 0.7, "zorder" : 100}
        default_marker_args = {"c" : "limegreen", "marker" : "+", "linewidth" : 15, "s" : 3, "zorder" : 100}
        
        line_args = {**default_line_args, **linesettings}
        marker_args = {**default_marker_args, **markersettings}

        ax.plot(self["R_leg_spline"], self["Z_leg_spline"], **line_args, label = self.name)
        ax.scatter(self["R_control"], self["Z_control"], **marker_args)

        ax.set_xlabel("$R\ (m)$")
        ax.set_ylabel("$Z\ (m)$")
        
        if ylim != (None,None):
            ax.set_ylim(ylim)
        if xlim != (None,None):
            ax.set_xlim(xlim)

        ax.set_title("RZ Space")
        ax.grid(alpha = 0.3, color = "k")
        ax.set_aspect("equal")