from netCDF4 import Dataset
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import os, sys
import pickle as pkl
import scipy as sp


class Morph():
    
    
    
    def __init__(self, R, Z, Xpoint, Btot, Bpol, S, Spol):
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
        


    def set_start_profile(self, offsets):
        self.start = self._set_profile(offsets)
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
        self.end = self._set_profile(offsets)
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
        
        
        
    def _set_profile(self, offsets):
        prof = {}
        prof["x"], prof["y"] = shift_points(self.R_leg, self.Z_leg, offsets)    # Points defining profile
        prof["xs"], prof["ys"] = cord_spline(prof["x"],prof["y"])   # Interpolate
        
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
        Btot_leg = start["Btot"][:start["Xpoint"]+1]
        Btot_leg_new = Btot_leg * (R_leg_new / start["R_leg"])
        
        prof["Btot"] = np.concatenate([
            Btot_leg_new,
            start["Btot"][start["Xpoint"]+1:], 
            ])
        
        prof["S"] = returnS(prof["R"], prof["Z"], prof["Btot"], prof["Bpol"])
        
        return prof
    
    
    
    def plot_profile(self, prof):
        
        fig, ax = plt.subplots(1, figsize = (6,12))
        
        s = self.start
        p = prof

        ax.plot(s["xs"], s["ys"], c = "forestgreen", zorder = 100, alpha = 1)
        ax.scatter(s["x"], s["y"], c = "limegreen", zorder = 100, marker = "+", linewidth = 15, s = 3)
        ax.plot(p["xs"], p["ys"], c = "deeppink", zorder = 100, alpha = 0.4)
        ax.scatter(p["x"], p["y"], c = "red", zorder = 100, marker = "x")

        ax.plot(s["R"], s["Z"], linewidth = 3, marker = "o", markersize = 0, color = "black", alpha = 1)
        # ax.plot(d_outer["R"], d_outer["Z"]*-1, linewidth = 3, marker = "o", markersize = 0, color = "black", alpha = 1)
        ax.set_xlabel("$R\ (m)$", fontsize = 15)
        ax.set_ylabel("$Z\ (m)$")
        ax.set_ylim(-8.8, -5.5)
        # ax.set_xlim(1.55, 2.7)

        alpha = 0.5
        ax.set_title("RZ Space")
        ax.grid(alpha = 0.3, color = "k")
        ax.set_aspect("equal")
        
        
        
    def plot_profile_check(self, prof):
        """
        Compare a new profile's Btot, Bpol, Spar, Spol and R,Z against the old one
        """
        fig, axes = plt.subplots(2,2, figsize = (8,8))

        d = self.start
        p = prof
        
        profstyle = dict(marker = "o", alpha = 0.3, c = "darkorange")
        xstyle = dict(marker = "+", linewidth = 2, s = 150, c = "r", zorder = 100)
        S_shift = p["S"][p["Xpoint"]] - d["S"][d["Xpoint"]] 
        Spol_shift = p["Spol"][p["Xpoint"]] - d["Spol"][d["Xpoint"]]
        
        
        ax = axes[0,0]
        ax.set_title("Total field (parallel)")
        
        ax.plot(d["S"] + S_shift, d["Btot"])
        ax.scatter(d["S"][d["Xpoint"]] + S_shift, d["Btot"][d["Xpoint"]], **xstyle)
        ax.plot(p["S"], p["Btot"], **profstyle)
        ax.scatter(p["S"][p["Xpoint"]], p["Btot"][p["Xpoint"]], **xstyle)
        ax.set_xlabel("Spar [m]");   ax.set_ylabel("B [T]")

        ax = axes[1,0]
        ax.set_title("Total field")
        
        ax.plot(d["Spol"] + Spol_shift, d["Btot"])
        ax.scatter(d["Spol"][d["Xpoint"]] + Spol_shift, d["Btot"][d["Xpoint"]], **xstyle)
        ax.plot(p["Spol"], p["Btot"], **profstyle)
        ax.scatter(p["Spol"][p["Xpoint"]], p["Btot"][p["Xpoint"]], **xstyle)
        ax.set_xlabel("Spol [m]");   ax.set_ylabel("B [T]")

        ax = axes[0,1]
        ax.set_title("R, Z space")
        
        ax.plot(d["R"], d["Z"])
        ax.scatter(d["R"][d["Xpoint"]], d["Z"][d["Xpoint"]], **xstyle)
        ax.plot(p["R"], p["Z"], **profstyle)
        ax.scatter(p["R"][p["Xpoint"]], p["Z"][p["Xpoint"]], **xstyle)
        ax.set_xlabel("R");   ax.set_ylabel("Z")

        ax = axes[1,1]
        ax.set_title("Poloidal field")
        
        ax.plot(d["Spol"] + Spol_shift, d["Bpol"])
        ax.scatter(d["Spol"][d["Xpoint"]] + Spol_shift,  (d["Bpol"])[d["Xpoint"]], **xstyle)
        ax.plot(p["Spol"], p["Bpol"], **profstyle)
        ax.scatter(p["Spol"][p["Xpoint"]],  (p["Bpol"])[p["Xpoint"]], **xstyle)
        ax.set_xlabel("Spol [m]");   ax.set_ylabel("B [T]")

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


def shift_points(R, Z, offsets):
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
    """
    
    #        XPOINT ---------   TARGET
    spl = cord_spline(R,Z, return_spline=True)
    x, y = [], []
    
    
    
    for i, point in enumerate(offsets):
        
        position = point["pos"]
        offsetx = point["offsetx"] if "offsetx" in point else 0
        offsety = point["offsety"] if "offsety" in point else 0
        
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