from netCDF4 import Dataset
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from collections import defaultdict

from freegs import critical
from freegs.equilibrium import Equilibrium
from freegs.machine import Wall
from freegs import jtor
from freegs import machine
from freegs.gradshafranov import mu0
from freegs import fieldtracer


from scipy import interpolate
import math
from numpy import (
linspace,
reshape,
ravel,
zeros,
clip,
)
from freegs.shaped_coil import ShapedCoil
from freegs.coil import Coil
from freegs.machine import Machine


from shapely.geometry import  LineString
from freegs.machine import Wall
from scipy.integrate import romb

def unpackConfigurationMK(File, 
                          Type, 
                          polModulator = 1, 
                          sepadd = 0, 
                          resolution = 300, 
                          convention = "target_to_midplane", 
                          filetype="balance",
                          diagnostic_plot = False,
                          absolute_B = True,
                          log_grid = False,
                          wallFile=False):
    """
    Extract interpolated variables along the SOL for connected double null configurations
    File = file path to either balance or eqdsk
    Type = iu, il, ou, ol, box: inner upper and lower, outer upper and lower, or slab geometry
    polModulator: multiplier on poloidal B field
    sepadd: code returns the nth sol ring from the separatrix, where n is sepadd
    convention: target_to_midplane has target at s=0, midplane_to_target has midplane at s=0
    filetype: either "balance" or "eqdsk"
    diagnostic_plot: plot a figure for a visual check
    absolute_B: return Bpol and Btot as absolute values
    wallFile: path of wall txt file in the case of eqdsk, else False

    Outputs:
    Bpol: Poloidal B field
    Btot: Total B field
    R: R coordinate along SOL segment/side
    Z: Z coordinates along SOL segment/side
    Xpoint: index of Xpoint in Bpol, Btot, R, Z, etc.
    Bx: Total B field at Xpoint
    zl: coordinates in Z space along SOL segment/side
    zx: Xpoint z coordinate
    Spol: poloidal distance
    S: parallel distance
    R_full: R coordinate of all cell centres in the grid
    Z_full: Z coordinate of all cell centres in the grid
    R_ring: R coordinate of all cell centres in the chosen SOL ring
    Z_ring: Z coordinate of all cell centres in the chosen SOL ring
    """

    """------DATA EXTRACTION"""
    full = dict() # dictionary to store parameters over full SOL ring
    if filetype == "balance":
        rootgrp = Dataset(File, "r", format="NETCDF4")
        sep = rootgrp['jsep'][0] # separatrix cell ring
        sep = sep + sepadd # select cell ring to work on
        bb = rootgrp["bb"] # B vector array

        full["Bpol"] = bb[0][sep]*polModulator

        # bb[A] returns B components x, y, z and B magnitude for 
        # A = 0, 1, 2 and 3 respectively for each cell centre in 2D grid
        full["Btot"] = bb[3][sep] # Mistake in Cyd's version. [3] is the total field
        
        # unpack dimensions of super-x
        # both r and z have the same shape as bb
        r = rootgrp['crx'] # corner radial coordinate (m)
        z = rootgrp['cry'] # corner vertical coordinate (m)

        # ring cell centres
        full["R"] = np.mean([r[0][sep], r[1][sep], r[2][sep], r[3][sep]], axis = 0)
        full["Z"] = np.mean([z[0][sep], z[1][sep], z[2][sep], z[3][sep]], axis = 0)

        # Entire grid cell centres
        Zs = np.mean([z[0], z[1], z[2], z[3]], axis = 0)
        Rs = np.mean([r[0], r[1], r[2], r[3]], axis = 0)
    else:
        full["R"],full["Z"],full["Btot"],full["Bpol"] = returnSOLring(File,wallFile)
        Zs = full["Z"]
        Rs = full["R"]

    len_R = len(full["R"])

    """------XPOINT, EDGE, MIDPLANE LOCATIONS"""
    # Find the X-points, edges and midpoints
    # SOLPS draws grid clockwise from inner lower target
    # Points of gradient flipping represent features of interest
    
    gradR = np.gradient(full["R"])
    
    reversals = []
    for i in range(1, len(gradR)):
        if np.sign(gradR[i-1]) != np.sign(gradR[i]):
            reversals.append(i)
    # Sometimes there are points of zero R gradient which result in duplicate reversals
    toremove = []
    for i in reversals:
        if gradR[i] == 0:
            toremove.append(i+1)
    for i in toremove:
        reversals.remove(i)
    
    omp = reversals[-2] # outer midplane
    imp = reversals[1] # inner midplane
    xpoint = dict()
    target = dict()
    sol = defaultdict(dict)

    xpoint["il"] = reversals[0] # inner lower xpoint
    xpoint["iu"] = reversals[2] # outer upper xpoint
    xpoint["ol"] = reversals[-1]-1 # outer lower xpoint
    xpoint["ou"] = reversals[-3] # outer upper xpoint

    target["il"] = 0
    target["iu"] = reversals[3]+1
    target["ou"] = reversals[4]-1
    target["ol"] = len(gradR)

    # Define start and end for each segment, going clockwise from bottom left.
    start = dict(); end = dict()
    start["il"] = target["il"]
    start["iu"] = imp-1
    start["ou"] = target["ou"]
    start["ol"] = omp-1

    end["il"] = imp+1
    end["iu"] = target["iu"]
    end["ou"] = omp+1
    end["ol"] = target["ol"]

    # Extract the four SOLs using their starts and ends.
    
    sol = defaultdict(dict) # Dict of parameters in each SOL side. sol[param][side]

    for param in full.keys():
        for side in ["il", "iu", "ou", "ol"]:
            sol[param][side] = full[param][start[side] : end[side]]

    # Invert inner lower and outer upper so that all SOLs 
    # are consistent and start at midplane (needs to be done under this convention to work)
    # This can be later reversed to start at target by input flag "convention=target_to_midplane"
    for param in full.keys():
        sol[param]["il"] = sol[param]["il"][::-1]
        sol[param]["ou"] = sol[param]["ou"][::-1]


    """------INTERPOLATION"""
    
    path_actual = dict() # Spol
    path_grid = dict() # grid to interpolate over
    interp = defaultdict(dict) # dict of interpolators
    data = defaultdict(dict) # final parameters
     
    for side in ["ol", "il", "ou", "iu"]:
        path_actual[side] = returnll(sol["R"][side], sol["Z"][side])

    # Find length between the two midplane cells so that we can interpolate from in-between them
    # Done by looking at first path length of the inner upper and final length of the outer upper
    # This works because the cells are in a clockwise order, therefore il -> iu -> ou -> ol
    imp_len = path_actual["iu"][1] - path_actual["iu"][0]
    omp_len = path_actual["ou"][1] - path_actual["ou"][0]

    # Offset the actual paths so they're zero at the actual midplane inbetween the two cells there
    # Then make grids to interpolate on that start at zero and end at the path end
    for side in ["iu", "il"]:
        path_actual[side] -= imp_len/2
        # path_grid[side] = np.linspace(0, np.amax(path_actual[side]), resolution)

    for side in ["ou", "ol"]:
        path_actual[side] -= omp_len/2
        # path_grid[side] = np.linspace(0, np.amax(path_actual[side]), resolution)

    # Create interpolators and apply them
    for side in ["ol", "il", "ou", "iu"]:
        
        # Regular linearly spaced grid (linear in poloidal space)
        path_grid[side] = np.linspace(0, np.amax(path_actual[side]), resolution)
        
        for param in full.keys():
            interp[side][param] = interpolate.interp1d(path_actual[side], sol[param][side], kind = "cubic")
            data[side][param] = interp[side][param](path_grid[side])

    # Find Xpoints again in the interpolated grid using gradient sign change
    # Stores xpoint index valid for each local divertor SOL space
    for side in ["ol", "il", "ou", "iu"]:
        gradientR = np.gradient(data[side]["R"])

        for i in range(len(gradientR)-1):
            if np.sign(gradientR[i]) != np.sign(gradientR[i+1]):
                data[side]["Xpoint"] = i+1

            

    """------OTHER CALCULATIONS"""
    
    zl = dict() # Z space path
    Bx = dict() # Btot at Xpoint
    polLengthArray = dict() # another poloidal distance path but now of the interpolated SOLs
    S = dict() # Real distance path as opposed to poloidal distance
    
    for side in ["ol", "il", "ou", "iu"]:
        d = data[side]
        
        # Reverse data if we want it target to midplane
        if convention == "target_to_midplane":
            for param in (x for x in d.keys() if x not in ["Xpoint"]):
                d[param] = d[param][::-1]
            d["Xpoint"] = len(d["R"]) - d["Xpoint"] - 1
            
        # Make B fields positive if required
        if absolute_B:
            d["Btot"] = abs(d["Btot"])
            d["Bpol"] = abs(d["Bpol"])
        
        d["Spol"] = np.array(returnll(d["R"], d["Z"]))
        
        if log_grid == True:
        # Create log grid from Xpoint to target in Spol space. Half the res goes above and half
        # goes below the Xpoint. This is much easier to do after the transformations above
        # because Spol and Xpoints exist and they're all in the right convention.

            Xpoint = d["Xpoint"]
            dymin = 0.001 # size of cell at the target
            abovex = np.linspace(d["Spol"][Xpoint], d["Spol"][-1],
                                 int(np.floor(resolution/2)))
            belowx = np.logspace(np.log10(dymin), np.log10(d["Spol"][Xpoint]),
                                 int(np.ceil(resolution/2)))
            belowx = np.insert(belowx, 0, 0) # logspace won't go to 0. Ensure there is one
            belowx = np.delete(belowx, -1) # Delete duplicate point shared with abovex
            path_grid_log = np.concatenate([belowx, abovex])
            
            # STILL OKAY AT THIS POINT

            # Reinterpolate the grid and redo all the transformations
            for param in full.keys():
                loginterp = interpolate.interp1d(d["Spol"], d[param])
                d[param] = loginterp(path_grid_log)
     
            d["Spol"] = path_grid_log
            d["Xpoint"] = len(belowx)
            
        
        # Assemble all the output variables
        d["Bx"] = d["Btot"][d["Xpoint"]]
        d["zl"] = np.array(returnzl(d["R"], d["Z"], d["Bx"], np.absolute(d["Bpol"])))
        d["zx"] = d["zl"][d["Xpoint"]]   
        d["S"] = np.array(returnS(d["R"], d["Z"], d["Btot"], d["Bpol"]))
        d["Sx"] = d["S"][d["Xpoint"]]
        d["Spolx"] = d["Spol"][d["Xpoint"]]

        # Full arrays of R and Z
        d["R_full"] = Rs
        d["Z_full"] = Zs
        d["R_ring"] = full["R"]
        d["Z_ring"] = full["Z"]
        

    """------OUTPUT"""    
    
    # Output by geometry type
    if Type != "box":
        return data[Type]
    else:
        print("Slab geometry not supported yet")

        
    """------DIAGNOSTIC PLOT""" 
    
    # Plot the four divertor SOLs and corresponding Xpoints
    if diagnostic_plot == True:
        
        fig, ax = plt.subplots(1,4, figsize = (18,4))

        xparam = "S"
        yparam = "Btot"

        for i, side in enumerate(["iu", "il", "ou", "ol"]):

            d = data[side]
            Xpoint = d["Xpoint"]
            ax[i].set_title(side)
            ax[i].plot(d[xparam], d[yparam], marker = "o", color = "None", markerfacecolor = "None", markeredgecolor = "purple", alpha = 0.5, zorder = 0)
            ax[i].scatter(d[xparam][Xpoint], d[yparam][Xpoint], marker = "o", s = 50, edgecolor = "black", color = "yellow", label = "Xpoint", zorder = 1)
            ax[i].legend(); ax[i].grid()
            ax[i].set_xlabel(xparam); ax[i].set_ylabel(yparam)
        
        fig.show()
        
        
    
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

def returnzl(R,Z, BX, Bpol):
    # return the distance in z from the target for a given configuration
    PrevR = R[0]
    PrevZ = Z[0]
    CurrentZ = 0
    zl = []
    for i in range(len(R)):
        dl = np.sqrt((PrevR-R[i])**2 + (PrevZ-Z[i])**2)
        dz = dl*BX/(Bpol[i])
        CurrentZ = CurrentZ+ dz
        zl.append(CurrentZ)
        PrevR = R[i]
        PrevZ = Z[i]
    return zl


def isPow2(val):
    """
    Returns True if val is a power of 2
    val     - Integer
    """
    return val & (val - 1) == 0


def ceilPow2(val):
    """
    Find a power of two greater than or equal to val
    """
    return 2 ** math.ceil(math.log2(val))


def readSeparatrix(
    fh,
    tokamak,
    cocos=1,
    wall=0,
    SOlmultiplier=0.0001,

):

    """
    Reads a G-EQDSK format file

    fh : File handle
    tokamak : Machine object defining coil locations
    cocos : integer
        COordinate COnventions. Not fully handled yet,
        only whether psi is divided by 2pi or not.
        if < 10 then psi is divided by 2pi, otherwise not.


    Returns
    -------

    R,Z, and interpolated field strengths along the separatrix

    """


    # Read the data as a Dictionary
    from freeqdsk import geqdsk ,aeqdsk
    data = geqdsk.read(fh, cocos=cocos,)
    


    # If data contains a limiter, set the machine wall
    if "rlim" in data:
        if len(data["rlim"]) > 3:
            tokamak.wall = Wall(data["rlim"], data["zlim"])
        else:
            print("Fewer than 3 points given for limiter/wall. Ignoring.")

    nx = data["nx"]
    ny = data["ny"]
    psi = data["psi"]

    if not (isPow2(nx - 1) and isPow2(ny - 1)):


        rin = linspace(0, 1, nx)
        zin = linspace(0, 1, ny)
        psi_interp = interpolate.RectBivariateSpline(rin, zin, psi, kx=1, ky=1)

        # Ensure that newnx, newny is 2^n + 1
        nx = ceilPow2(nx - 1) + 1
        ny = ceilPow2(ny - 1) + 1


        rnew = linspace(0, 1, nx)
        znew = linspace(0, 1, ny)

        # Interpolate onto new grid
        psi = psi_interp(rnew, znew)

    # Range of psi normalises psi derivatives
    psi_bndry = data["sibdry"]
    psi_axis = data["simagx"]
    psirange = data["sibdry"] - data["simagx"]

    psinorm = linspace(0.0, 1.0, data["nx"], endpoint=True)

    # Create a spline fit to pressure, f and f**2
    p_spl = interpolate.InterpolatedUnivariateSpline(psinorm, data["pres"])
    pprime_spl = interpolate.InterpolatedUnivariateSpline(
        psinorm, data["pres"] / psirange
    ).derivative()

    f_spl = interpolate.InterpolatedUnivariateSpline(psinorm, data["fpol"])
    ffprime_spl = interpolate.InterpolatedUnivariateSpline(
        psinorm, 0.5 * data["fpol"] ** 2 / psirange
    ).derivative()

    q_spl = interpolate.InterpolatedUnivariateSpline(psinorm, data["qpsi"])

    # functions to return p, pprime, f and ffprime
    def p_func(psinorm):
        if hasattr(psinorm, "shape"):
            return reshape(p_spl(ravel(psinorm)), psinorm.shape)
        return p_spl(psinorm)

    def f_func(psinorm):
        if hasattr(psinorm, "shape"):
            return reshape(f_spl(ravel(psinorm)), psinorm.shape)
        return f_spl(psinorm)

    def pprime_func(psinorm):
        if hasattr(psinorm, "shape"):
            return reshape(pprime_spl(ravel(psinorm)), psinorm.shape)
        return pprime_spl(psinorm)

    def ffprime_func(psinorm):
        if hasattr(psinorm, "shape"):
            return reshape(ffprime_spl(ravel(psinorm)), psinorm.shape)
        return ffprime_spl(psinorm)

    def q_func(psinorm):
        if hasattr(psinorm, "shape"):
            return reshape(q_spl(ravel(psinorm)), psinorm.shape)
        return q_spl(psinorm)

    # Create a set of profiles to calculate toroidal current density Jtor
    profiles = jtor.ProfilesPprimeFfprime(
        pprime_func,
        ffprime_func,
        data["rcentr"] * data["bcentr"],
        p_func=p_func,
        f_func=f_func,
    )

    # Calculate normalised psi.
    # 0 = magnetic axis
    # 1 = plasma boundary
    psi_norm = clip((psi - psi_axis) / (psi_bndry - psi_axis), 0.0, 1.1)

    # Create an Equilibrium object
    eq = Equilibrium(
        tokamak=tokamak,
        Rmin=data["rleft"],
        Rmax=data["rleft"] + data["rdim"],
        Zmin=data["zmid"] - 0.5 * data["zdim"],
        Zmax=data["zmid"] + 0.5 * data["zdim"],
        nx=nx,
        ny=ny,  # Number of grid points

    )
    eq._updatePlasmaPsi(data["psi"])

    # Get profiles, particularly f needed for toroidal field

    from numpy import concatenate,  reshape, ravel

    psinorm = linspace(0.0, 1.0, data["nx"], endpoint=True)
    f_spl = interpolate.InterpolatedUnivariateSpline(psinorm, data["fpol"])
    def f_func(psinorm):
        if hasattr(psinorm, "shape"):
            return reshape(f_spl(ravel(psinorm)),psinorm.shape)
        return f_spl(psinorm)

    eq._profiles = jtor.ProfilesPprimeFfprime(None,
                                                None,
                                                data["rcentr"] * data["bcentr"],
                                                f_func=f_func)
    
    # Set a wall 
    eq.tokamak.wall =wall

    # Find all the O- and X-points
    opoint, xpoint = critical.find_critical(eq.R, eq.Z, psi)


    # Draw separatrix if there is an X-point
    diff = xpoint[0][2]-opoint[0][2]
    if len(xpoint) > 0:
        cs = plt.contour(eq.R, eq.Z, psi, levels=[xpoint[0][2]+SOlmultiplier*diff], colors="r")
    p = cs.collections[0].get_paths()[0]
    v = p.vertices
    x = v[:,0]
    y = v[:,1]
    outerx = 0
    outery = 0

    plt.close()

    if len(cs.collections[0].get_paths())>1:
        p2 = cs.collections[0].get_paths()[1]
        v2 = p2.vertices
        outerx = v2[:,0]
        outery = v2[:,1]

    return x, y,outerx,outery, eq.Btot, eq.Br, eq.Bz



def returnSOLring(eqname,wallname,SOlmultiplier=0.0001):
    """
    Unpacks separatrix data

    eqname : Location of geqdsk file
    wallname : Location of wall txt file
    SOlmultiplier : larger number means the RING is further into the SOL


    Returns
    -------

    R,Z,Btot and Bpol along a SOL ring 

    """
    from shapely.geometry import Point
    from shapely.geometry.polygon import Polygon

    # read wall file
    wallContour = np.loadtxt(wallname)
    wallContour = wallContour
    wallpolygon = []
    for i in range(len(wallContour[:,0])):
        wallpolygon.append((wallContour[:,0][i],wallContour[:,1][i]))
    polygon = Polygon(wallpolygon)
    wall = Wall(
    wallContour[:,0], wallContour[:,1]  # R
    )  

    # open equilibrium file
    with open(eqname) as f:

        tokamak = machine.EmptyTokamak()

        # read separatrix
        R,Z,Router,Zouter,btot,br,bz = readSeparatrix(f,tokamak=tokamak,wall=wall,SOlmultiplier=SOlmultiplier)

        Rkeep = []
        Zkeep = []
        Btotkeep = []
        Bpolkeep = []
        dist = []

        # keep points within wall domain
        for i in range(len(R)):
            point = Point(R[i],Z[i])
            if polygon.contains(point):
                Rkeep.append(R[i])
                Zkeep.append(Z[i])

        # find start index
        for i in range(len(Rkeep)):
            dist.append((Rkeep[i]-Rkeep[i-1])**2+(Zkeep[i]-Zkeep[i-1])**2)
        startind = np.argmax(dist)
        Rkeep = np.append(Rkeep[startind:],Rkeep[:startind])
        Zkeep = np.append(Zkeep[startind:],Zkeep[:startind])

        #add outer if equilibrium is double null
        if isinstance(Router, int) and Rkeep[0]>Rkeep[-1]:
            Rkeep = Rkeep[::-1]
            Zkeep = Zkeep[::-1]
        else:
            if Zkeep[0]>Zkeep[-1]:
                Rkeep = Rkeep[::-1]
                Zkeep = Zkeep[::-1]

            # add the outer divertor
            Rkeepouter= []
            Zkeepouter = []
            for i in range(len(Router)):
                point = Point(Router[i],Zouter[i])

                if polygon.contains(point):
                    Rkeepouter.append(Router[i])
                    Zkeepouter.append(Zouter[i])
                
            # ensure it starts at upper outer
            dist = []
            for i in range(len(Rkeepouter)):
                dist.append((Rkeepouter[i]-Rkeepouter[i-1])**2+(Zkeepouter[i]-Zkeepouter[i-1])**2)
            startind = np.argmax(dist)
            Rkeepouter = np.append(Rkeepouter[startind:],Rkeepouter[:startind])
            Zkeepouter = np.append(Zkeepouter[startind:],Zkeepouter[:startind])
            if Zkeepouter[0]<Zkeepouter[-1]:
                Rkeepouter = Rkeepouter[::-1]
                Zkeepouter = Zkeepouter[::-1]
            Rkeep = np.append(Rkeep,Rkeepouter)
            Zkeep = np.append(Zkeep,Zkeepouter)

        # add magnetic field profiles
        for i in range(len(Rkeep)):
            bpolpoint = np.sqrt(br(Rkeep[i],Zkeep[i])**2+bz(Rkeep[i],Zkeep[i])**2)
            btotpoint = np.sqrt(btot(Rkeep[i],Zkeep[i])**2)
            Btotkeep.append(btotpoint)
            Bpolkeep.append(bpolpoint)

    return Rkeep,Zkeep,Btotkeep,Bpolkeep
