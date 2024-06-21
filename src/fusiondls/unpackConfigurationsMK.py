from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from scipy import interpolate

from .Profile import Profile


def unpackConfigurationMK(
    File,
    Type,
    polModulator=1,
    sepadd=0,
    resolution=300,
    convention="target_to_midplane",
    diagnostic_plot=False,
    absolute_B=True,
):
    """
    Extract interpolated variables along the SOL for connected double null configurations
    File = balance file path
    Type = iu, il, ou, ol, box: inner upper and lower, outer upper and lower, or slab geometry
    polModulator: multiplier on poloidal B field
    sepadd: code returns the nth sol ring from the separatrix, where n is sepadd
    convention: target_to_midplane has target at s=0, midplane_to_target has midplane at s=0
    diagnostic_plot: plot a figure for a visual check
    absolute_B: return Bpol and Btot as absolute values

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
    rootgrp = Dataset(File, "r", format="NETCDF4")
    sep = rootgrp["jsep"][0]  # separatrix cell ring
    sep += sepadd  # select cell ring to work on
    bb = rootgrp["bb"]  # B vector array

    full = {}  # dictionary to store parameters over full SOL ring

    full["Bpol"] = bb[0][sep] * polModulator

    # bb[A] returns B components x, y, z and B magnitude for
    # A = 0, 1, 2 and 3 respectively for each cell centre in 2D grid
    full["Btot"] = bb[3][sep]  # Mistake in Cyd's version. [3] is the total field

    # unpack dimensions of super-x
    # both r and z have the same shape as bb
    r = rootgrp["crx"]  # corner radial coordinate (m)
    z = rootgrp["cry"]  # corner vertical coordinate (m)

    # ring cell centres
    full["R"] = np.mean([r[0][sep], r[1][sep], r[2][sep], r[3][sep]], axis=0)
    full["Z"] = np.mean([z[0][sep], z[1][sep], z[2][sep], z[3][sep]], axis=0)

    """------XPOINT, EDGE, MIDPLANE LOCATIONS"""
    # Find the X-points, edges and midpoints
    # SOLPS draws grid clockwise from inner lower target
    # Points of gradient flipping represent features of interest

    gradR = np.gradient(full["R"])

    reversals = []
    for i in range(1, len(gradR)):
        if np.sign(gradR[i - 1]) != np.sign(gradR[i]):
            reversals.append(i)

    omp = reversals[-2]  # outer midplane
    imp = reversals[1]  # inner midplane
    xpoint = {}
    target = {}
    sol = defaultdict(dict)

    xpoint["il"] = reversals[0]  # inner lower xpoint
    xpoint["iu"] = reversals[2]  # outer upper xpoint
    xpoint["ol"] = reversals[-1] - 1  # outer lower xpoint
    xpoint["ou"] = reversals[-3]  # outer upper xpoint

    target["il"] = 0
    target["iu"] = reversals[3] + 1
    target["ou"] = reversals[4] - 1
    target["ol"] = len(gradR)

    if diagnostic_plot:
        fig, ax = plt.subplots(dpi=150)
        ax.set_aspect("equal")
        ax.scatter(full["R"], full["Z"], s=5)
        for i in reversals:
            ax.scatter(full["R"][i], full["Z"][i], color="red", marker="*", s=10)

    # Define start and end for each segment, going clockwise from bottom left.
    start = {}
    end = {}
    start["il"] = target["il"]
    start["iu"] = imp - 1
    start["ou"] = target["ou"]
    start["ol"] = omp - 1

    end["il"] = imp + 1
    end["iu"] = target["iu"]
    end["ou"] = omp + 1
    end["ol"] = target["ol"]

    # Extract the four SOLs using their starts and ends.

    sol = defaultdict(dict)  # Dict of parameters in each SOL side. sol[param][side]

    for param in full:
        for side in ["il", "iu", "ou", "ol"]:
            sol[param][side] = full[param][start[side] : end[side]]

    # Invert inner lower and outer upper so that all SOLs
    # are consistent and start at midplane (needs to be done under this convention to work)
    # This can be later reversed to start at target by input flag "convention=target_to_midplane"
    for param in full:
        sol[param]["il"] = sol[param]["il"][::-1]
        sol[param]["ou"] = sol[param]["ou"][::-1]

    """------INTERPOLATION"""

    path_actual = {}  # Spol
    path_grid = {}  # grid to interpolate over
    interp = defaultdict(dict)  # dict of interpolators
    data = defaultdict(dict)  # final parameters

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
        path_actual[side] -= imp_len / 2
        # path_grid[side] = np.linspace(0, np.amax(path_actual[side]), resolution)

    for side in ["ou", "ol"]:
        path_actual[side] -= omp_len / 2
        # path_grid[side] = np.linspace(0, np.amax(path_actual[side]), resolution)

    # Create interpolators and apply them
    for side in ["ol", "il", "ou", "iu"]:
        # Regular linearly spaced grid (linear in poloidal space)
        path_grid[side] = np.linspace(0, np.amax(path_actual[side]), resolution)

        for param in full:
            interp[side][param] = interpolate.interp1d(
                path_actual[side], sol[param][side], kind="cubic"
            )
            data[side][param] = interp[side][param](path_grid[side])

    # Find Xpoints again in the interpolated grid using gradient sign change
    # Stores xpoint index valid for each local divertor SOL space
    for side in ["ol", "il", "ou", "iu"]:
        gradientR = np.gradient(data[side]["R"])

        for i in range(len(gradientR) - 1):
            if np.sign(gradientR[i]) != np.sign(gradientR[i + 1]):
                data[side]["Xpoint"] = i + 1

    """------OTHER CALCULATIONS"""

    for side in ["ol", "il", "ou", "iu"]:
        d = data[side]

        # Reverse data if we want it target to midplane
        if convention == "target_to_midplane":
            for param in (x for x in d if x != "Xpoint"):
                d[param] = d[param][::-1]
            d["Xpoint"] = len(d["R"]) - d["Xpoint"] - 1

        # Make B fields positive if required
        if absolute_B:
            d["Btot"] = abs(d["Btot"])
            d["Bpol"] = abs(d["Bpol"])

        # Poloidal distance
        d["Spol"] = np.array(returnll(d["R"], d["Z"]))
        d["S"] = np.array(returnS(d["R"], d["Z"], d["Btot"], d["Bpol"]))

        # Z space distance (combined parallel and flux expansion, see Lipschultz 2016)
        d["zl"] = np.array(
            returnzl(d["R"], d["Z"], d["Btot"][d["Xpoint"]], np.absolute(d["Bpol"]))
        )

    """------DIAGNOSTIC PLOT"""

    # Plot the four divertor SOLs and corresponding Xpoints
    if diagnostic_plot:
        fig, ax = plt.subplots(1, 4, figsize=(18, 4))

        xparam = "S"
        yparam = "Btot"

        for i, side in enumerate(["iu", "il", "ou", "ol"]):
            d = data[side]
            Xpoint = d["Xpoint"]
            ax[i].set_title(side)
            ax[i].plot(
                d[xparam],
                d[yparam],
                marker="o",
                color="None",
                markerfacecolor="None",
                markeredgecolor="purple",
                alpha=0.5,
                zorder=0,
            )
            ax[i].scatter(
                d[xparam][Xpoint],
                d[yparam][Xpoint],
                marker="o",
                s=50,
                edgecolor="black",
                color="yellow",
                label="Xpoint",
                zorder=1,
            )
            ax[i].legend()
            ax[i].grid()
            ax[i].set_xlabel(xparam)
            ax[i].set_ylabel(yparam)

        fig.show()

    """------OUTPUT"""

    # Pack into a Profile class
    profiles = {}
    for side in ["iu", "il", "ou", "ol"]:
        d = data[side]
        profiles[side] = Profile(
            d["R"],
            d["Z"],
            d["Xpoint"],
            d["Btot"],
            d["Bpol"],
            d["S"],
            d["Spol"],
            name=side,
        )

    # Output by geometry type
    if Type != "box":
        return profiles[Type]

    print("Slab geometry not supported yet")
    return None


def returnll(R, Z):
    # return the poloidal distances from the target for a given configuration
    PrevR = R[0]
    ll = []
    currentl = 0
    PrevZ = Z[0]
    for i in range(len(R)):
        dl = np.sqrt((PrevR - R[i]) ** 2 + (PrevZ - Z[i]) ** 2)
        currentl += dl
        ll.append(currentl)
        PrevR = R[i]
        PrevZ = Z[i]
    return ll


def returnS(R, Z, B, Bpol):
    # return the real total distances from the target for a given configuration
    PrevR = R[0]
    s = []
    currents = 0
    PrevZ = Z[0]
    for i in range(len(R)):
        dl = np.sqrt((PrevR - R[i]) ** 2 + (PrevZ - Z[i]) ** 2)
        ds = dl * np.abs(B[i]) / np.abs(Bpol[i])
        currents += ds
        s.append(currents)
        PrevR = R[i]
        PrevZ = Z[i]
    return s


def returnzl(R, Z, BX, Bpol):
    # return the distance in z from the target for a given configuration
    PrevR = R[0]
    PrevZ = Z[0]
    CurrentZ = 0
    zl = []
    for i in range(len(R)):
        dl = np.sqrt((PrevR - R[i]) ** 2 + (PrevZ - Z[i]) ** 2)
        dz = dl * BX / (Bpol[i])
        CurrentZ += dz
        zl.append(CurrentZ)
        PrevR = R[i]
        PrevZ = Z[i]
    return zl
