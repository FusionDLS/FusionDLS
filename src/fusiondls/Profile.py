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
    Class defining a single field line profile (field line with topology).
    Used in the geometry reader. Has a lot of methods to calculate basic statistics
    as well as to modify the profile, e.g. to change its connection length or
    total flux expansion.
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



class Morph():
    """
    This class creates new field line profiles by interpolating between
    any two profiles. You provide a start and end profile and it will return
    intermediate ones according to a morph factor.
    """
    
    
    
    def __init__(self, R, Z, Xpoint, Btot, Bpol, S, Spol):
        """
        Class is initialised with the properties of the base (start) profile.
        Needs to be refactored to accept a Profile class.
        """
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
        
    

    def set_start_profile(self, profile, offsets):
        """
        Sets the start profile based on what the class was initialised with.
        You must provide offsets which is a dictionary of the spline control
        points and their offsets. For the start profile, the offsets dictionary
        should contain just the control points with no offsets.
        See the function shift_offsets() for what the offsets should look like.
        """
        self.start = self.make_profile_spline(profile, offsets)
        self.start["R_leg"] = self.R_leg
        self.start["Z_leg"] = self.Z_leg
        self.start["R"] = self.R
        self.start["Z"] = self.Z
        self.start["S"] = self.S
        self.start["Spol"] = self.Spol
        self.start["Btot"] = self.Btot
        self.start["Bpol"] = self.Bpol
        self.start["Xpoint"] = self.Xpoint
        
        
        
    def set_end_profile(self, offsets):
        """
        Sets the end profile based on the offsets dictionary with the original
        control point coordinates and their desired offsets.
        """
        self.end = self.make_profile_spline(offsets)
        self.end = self._populate_profile(self.end)
        
        
        
    def generate_profiles(self, factors):
        """ 
        Make a series of profiles according to provided factors
        where factor = 0 corresponds to start, factor = 1
        corresponds to end and factor = 0.5 corresponds to halfway.
        """
        profiles = {}
        for i in factors:
            profiles[i] = self.morph_between(i)
        
        self.profiles = profiles
        
        
        
    def morph_between(self, factor):
        
        prof = {}
        prof["x"] = self.start["x"] + factor*(self.end["x"] - self.start["x"])
        prof["y"] = self.start["y"] + factor*(self.end["y"] - self.start["y"])
        prof["xs"], prof["ys"] = cord_spline(prof["x"],prof["y"])   # Interpolate
        prof = self._populate_profile(prof)
        
        return prof
        
    
    
    def _populate_profile(self, prof):
        """ 
        Add the rest of the profile to the leg above the X-point
        Add Bpol and Btot along entire leg
        Returns new modified profile
        """
        
        start = self.start
        prof["Xpoint"] = start["Xpoint"]
        
        ## Add leg above X-point
        # xs and ys are backwards
        dist = get_cord_distance(start["R_leg"], start["Z_leg"])   # Distances along old leg
        spl = cord_spline(prof["xs"][::-1], prof["ys"][::-1], return_spline = True)   # Spline interp for new leg
        R_leg_new, Z_leg_new = spl(dist)     # New leg interpolated onto same points as old leg

        prof["R"] = np.concatenate([
            R_leg_new,
            start["R"][start["Xpoint"]+1:], 
            ])
        
        prof["Z"] = np.concatenate([
            Z_leg_new,
            start["Z"][start["Xpoint"]+1:], 
            ])

        ## Poloidal dist and field
        prof["Spol"] = returnll(prof["R"], prof["Z"])
        prof["Bpol"] = start["Bpol"].copy()    # Assume same poloidal field as start
        
        ## Total field 
        Btor = np.sqrt(start["Btot"]**2 - start["Bpol"]**2)   # Toroidal field
        Btor_leg = Btor[:start["Xpoint"]+1]
        Btor_leg_new = Btor_leg * (start["R_leg"] / R_leg_new)

        Bpol_leg = start["Bpol"][:start["Xpoint"]+1]
        Btot_leg_new = np.sqrt(Btor_leg_new**2 + Bpol_leg**2)
        
        prof["Btot"] = np.concatenate([
            Btot_leg_new,
            start["Btot"][start["Xpoint"]+1:], 
            ])
        
        prof["S"] = returnS(prof["R"], prof["Z"], prof["Btot"], prof["Bpol"])
        
        return prof
    
    
    
    def plot_profile(self, prof, dpi=100, ylim=(None,None), xlim=(None,None)):
        
        fig, ax = plt.subplots(dpi = dpi)
        
        s = self.start
        p = prof

        ax.plot(s["xs"], s["ys"], c = "forestgreen", zorder = 100, alpha = 1)
        ax.scatter(s["x"], s["y"], c = "limegreen", zorder = 100, marker = "+", linewidth = 15, s = 3)
        ax.plot(p["xs"], p["ys"], c = "deeppink", zorder = 100, alpha = 0.4)
        ax.scatter(p["x"], p["y"], c = "red", zorder = 100, marker = "x")

        ax.plot(s["R"], s["Z"], linewidth = 3, marker = "o", markersize = 0, color = "black", alpha = 1)
        
        # ax.plot(d_outer["R"], d_outer["Z"], linewidth = 3, marker = "o", markersize = 0, color = "black", alpha = 1)
        ax.set_xlabel("$R\ (m)$", fontsize = 15)
        ax.set_ylabel("$Z\ (m)$")
        
        if ylim != (None,None):
            ax.set_ylim(ylim)
        if xlim != (None,None):
            ax.set_xlim(xlim)

        alpha = 0.5
        ax.set_title("RZ Space")
        ax.grid(alpha = 0.3, color = "k")
        ax.set_aspect("equal")
        
        
        
def compare_profile_topologies(base_profile, profiles):
    """
    Do a bunch of plots to compare the properties of two profiles
    """

    d = base_profile
    
    fig, axes = plt.subplots(2,2, figsize = (8,8))
    markers = ["o", "v"]

    profstyle = dict(alpha = 0.3)
    

    basestyle = dict(c = "black")
    xstyle = dict(marker = "+", linewidth = 2, s = 150, c = "r", zorder = 100)

    S_xpoint_max = max([p["S"][p["Xpoint"]] for p in profiles])
    S_pol_xpoint_max = max([p["Spol"][p["Xpoint"]] for p in profiles])

    Spol_shift_base = S_pol_xpoint_max - d["Spol"][d["Xpoint"]] 



    ax = axes[0,0]
    ax.set_title("Fractional $B_{tot}$ gradient")

    ax.plot(d["Spol"] + Spol_shift_base, np.gradient(d["Btot"], d["Spol"]) / d["Btot"], **basestyle)
    ax.scatter(d["Spol"][d["Xpoint"]] + Spol_shift_base, (np.gradient(d["Btot"], d["Spol"]) / d["Btot"])[d["Xpoint"]], **xstyle)
    for i, p in enumerate(profiles): 
        Spol_shift = S_pol_xpoint_max  - p["Spol"][p["Xpoint"]]
        ax.plot(p["Spol"] + Spol_shift, np.gradient(p["Btot"], p["Spol"]) / p["Btot"], **profstyle, marker = markers[i])
        # ax.scatter(p["Spol"][p["Xpoint"]]+ Spol_shift, (np.gradient(p["Btot"], p["Spol"]) / p["Btot"])[p["Xpoint"]], **xstyle)
        ax.set_xlabel(r"$S_{\theta} \   [m]$");   
        ax.set_ylabel("$B_{tot}$ $[T]$")


    ax = axes[1,0]
    ax.set_title("$B_{tot}$")

    ax.plot(d["Spol"] + Spol_shift_base, d["Btot"], **basestyle)
    ax.scatter(d["Spol"][d["Xpoint"]] + Spol_shift_base, d["Btot"][d["Xpoint"]], **xstyle)
    for i, p in enumerate(profiles): 
        Spol_shift = S_pol_xpoint_max  - p["Spol"][p["Xpoint"]]
        ax.plot(p["Spol"] + Spol_shift, p["Btot"], **profstyle, marker = markers[i])
        ax.set_xlabel(r"$S_{\theta} \   [m]$")
        ax.set_ylabel("$B_{tot}$ $[T]$")


    ax = axes[0,1]
    ax.set_title(r"Field line pitch $B_{pol}/B_{tot}$")
    
    ax.plot(d["Spol"] + Spol_shift_base, d["Bpol"]/d["Btot"], **basestyle)
    ax.scatter(d["Spol"][d["Xpoint"]]+ Spol_shift_base, (d["Bpol"]/d["Btot"])[d["Xpoint"]], **xstyle)
    for i, p in enumerate(profiles): 
        Spol_shift = S_pol_xpoint_max  - p["Spol"][p["Xpoint"]]
        ax.plot(p["Spol"] + Spol_shift, p["Bpol"]/p["Btot"], **profstyle, marker = markers[i])
    ax.set_xlabel(r"$S_{\theta} \   [m]$")
    ax.set_ylabel(r"$B_{pol} \ / B_{tot}$ ")

    ax = axes[1,1]
    ax.set_title("$B_{pol}$")

    ax.plot(d["Spol"] + Spol_shift_base, d["Bpol"], **basestyle)
    ax.scatter(d["Spol"][d["Xpoint"]] + Spol_shift_base,  (d["Bpol"])[d["Xpoint"]], **xstyle)
    for i, p in enumerate(profiles): 
        Spol_shift = S_pol_xpoint_max  - p["Spol"][p["Xpoint"]]
        ax.plot(p["Spol"] + Spol_shift, p["Bpol"], **profstyle, marker = markers[i])
        ax.scatter(p["Spol"][p["Xpoint"]] + Spol_shift,  (p["Bpol"])[p["Xpoint"]], **xstyle)
    ax.set_xlabel(r"$S_{\theta} \   [m]$")
    ax.set_ylabel(r"$B_{\theta}$ $[T]$")


    fig.tight_layout()
    
    
    
def cord_spline(x,y, return_spline = False):
    """ 
    Do cord interpolation of x and y. This parametrises them
    by the cord length and allows them to go back on themselves, 
    i.e. to have non-unique X values and non-monotonicity.
    I think you need to have at least 4 points.
    
    https://docs.scipy.org/doc/scipy/tutorial/interpolate/1D.html#parametric-spline-curves
    """
    p = np.stack((x,y))
    u_cord = get_cord_distance(x,y)

    spl = sp.interpolate.make_interp_spline(u_cord, p, axis=1)

    uu = np.linspace(u_cord[0], u_cord[-1], 200)
    R, Z = spl(uu)
    
    if return_spline:
        return spl
    else:
        return R,Z
    
    
    
def get_cord_distance(x,y):
    """ 
    Return array of distances along a curve defined by x and y.
    """
    p = np.stack((x,y))
    dp = p[:,1:] - p[:,:-1]        # 2-vector distances between points
    l = (dp**2).sum(axis=0)        # squares of lengths of 2-vectors between points
    u_cord = np.sqrt(l).cumsum()   # Cumulative sum of 2-norms
    u_cord /= u_cord[-1]           # normalize to interval [0,1]
    u_cord = np.r_[0, u_cord]      # the first point is parameterized at zero
    
    return u_cord



def shift_points(R, Z, offsets, factor = 1):
    """ 
    Make control points on a field line according to points of index in list i.
    
    Parameters
    ----------
    R, Z: 1D arrays
        R and Z coordinates of field line.
    i: list of ints
        Indices of points to shift. They are the control points of the spline.
    yoffset: list of floats
        Y offset to apply to each point in i.
    xoffset: list of floats
        X offset to apply to each point in i.
    """
    
    #        XPOINT ---------   TARGET
    spl = cord_spline(R,Z, return_spline=True)
    x, y = [], []
    
    
    
    for i, point in enumerate(offsets):
        
        position = point["pos"]
        offsetx = point["offsetx"] if "offsetx" in point else 0
        offsety = point["offsety"] if "offsety" in point else 0
        
        offsetx *= factor
        offsety *= factor
        
        Rs, Zs = spl(position)
        x.append(Rs+offsetx)
        y.append(Zs+offsety)
        # x = [R[i[0]], R[i[1]], R[i[2]], R[i[3]]]
        # y = [Z[i[0]]+yoffset[0], Z[i[1]]+yoffset[1], Z[i[2]]+yoffset[2], Z[i[3]]+yoffset[3]]
    
    return np.array(x), np.array(y)
    
   
    
def returnll(R,Z):
    #return the poloidal distances from the target for a given configuration
    PrevR = R[0]
    ll = []
    currentl = 0
    PrevZ = Z[0]
    for i in range(len(R)):
        dl = np.sqrt((PrevR-R[i])**2 + (PrevZ-Z[i])**2)
        currentl = currentl+ dl
        ll.append(currentl)
        PrevR = R[i]
        PrevZ = Z[i]
    return ll



def returnS(R,Z,B,Bpol):
    #return the real total distances from the target for a given configuration
    PrevR = R[0]
    s = []
    currents = 0
    PrevZ = Z[0]
    for i in range(len(R)):
        dl = np.sqrt((PrevR-R[i])**2 + (PrevZ-Z[i])**2)
        ds = dl*np.abs(B[i])/np.abs(Bpol[i])
        currents = currents+ ds
        s.append(currents)
        PrevR = R[i]
        PrevZ = Z[i]
    return s