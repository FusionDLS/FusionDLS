import numpy as np
from netCDF4 import Dataset
from scipy import interpolate

# unpack kink fields


def find_nearest(array, value):
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()


def unpackConfiguration(
    File, Type, zxoverL=0, returnSBool=False, polModulator=1, sepadd=0
):
    rootgrp = Dataset(File, "r", format="NETCDF4")
    sep = rootgrp["jsep"][0]
    sep += sepadd
    bb = rootgrp["bb"]
    Bpol = bb[0][sep] * polModulator
    TotalField = np.sqrt(bb[3][sep] ** 2 + bb[0][sep] ** 2)
    # unpack dimensions of super-x
    r = rootgrp["crx"]
    z = rootgrp["cry"]

    rbl = r[0][sep]
    rbr = r[1][sep]
    rtl = r[2][sep]
    rtr = r[3][sep]

    zbl = z[0][sep]
    zbr = z[1][sep]
    ztl = z[2][sep]
    ztr = z[3][sep]

    Z = (zbl + zbr + ztl + ztr) / 4
    R = (rbl + rbr + rtl + rtr) / 4

    Zs = (z[0] + z[1] + z[2] + z[3]) / 4
    Rs = (r[0] + r[1] + r[2] + r[3]) / 4

    gradR = np.gradient(R)
    Xpoint = -1
    # cut off data such that it extens from the target to the x point
    if Type == "outer":
        Start = 0

        for i in range(1, len(gradR)):
            if np.sign(gradR[i - 1]) != np.sign(gradR[i]) and R[i] >= 1:
                Start = i - 1
                break

        R = R[Start:]
        Bpol = Bpol[Start:]
        gradR = np.gradient(R)

        Xpoint = 0
        midplane = 0

        for i in range(1, len(gradR)):
            if np.sign(gradR[i - 1]) != np.sign(gradR[i]):
                Xpoint = i - 1
                break
        for i in range(Xpoint + 2, len(gradR)):
            if np.sign(gradR[i - 1]) != np.sign(gradR[i]):
                midplane = i - 1
                break

        TotalField = TotalField[Start:]
        TotalField = TotalField[0 : midplane + 1]

        Z = Z[Start:]
        Z = -Z[0 : midplane + 1]
        R = R[0 : midplane + 1]
        Bpol = Bpol[0 : midplane + 1]

    if Type == "inner":
        Start = 0

        R = R[Start:]
        Bpol = Bpol[Start:]
        gradR = np.gradient(R)

        Xpoint = 0  # Ryoko
        midplane = 0  # Ryoko
        for i in range(1, len(gradR)):
            if np.sign(gradR[i - 1]) != np.sign(gradR[i]):
                Xpoint = i - 1
                break
        for i in range(Xpoint + 2, len(gradR)):
            if np.sign(gradR[i - 1]) != np.sign(gradR[i]):
                midplane = i - 1
                break
        TotalField = TotalField[Start:]
        TotalField = TotalField[0 : midplane + 1]

        print("Xpoint,midplane")
        print(Xpoint, midplane)
        Z = Z[Start : midplane + 1]
        R = R[0 : midplane + 1]
        Bpol = Bpol[0 : midplane + 1]

    if Type == "Box":
        Bpol = Bpol[::-1]
        TotalField = TotalField[::-1]
        R = R[::-1]
        Z = Z[::-1]

    pathLength = returnll(R, Z)

    # interpolate the grid to make it smooth

    Zinterp = interpolate.interp1d(pathLength, Z, kind="cubic")
    Rinterp = interpolate.interp1d(pathLength, R, kind="cubic")
    Bpolinterp = interpolate.interp1d(pathLength, Bpol, kind="cubic")
    TotalFieldinterp = interpolate.interp1d(pathLength, TotalField, kind="cubic")
    path = np.linspace(np.amin(pathLength), np.amax(pathLength), 10000)
    R = Rinterp(path)
    Z = Zinterp(path)
    Bpol = Bpolinterp(path)
    TotalField = TotalFieldinterp(path)
    gradR = np.gradient(R)
    for i in range(1, len(gradR)):
        if np.sign(gradR[i - 1]) != np.sign(gradR[i]):
            Xpoint = i - 1
            break

    Bx = TotalField[Xpoint]
    zl = np.array(returnzl(R, Z, Bx, np.absolute(Bpol)))
    if Type == "Box":
        Xpoint = find_nearest(zl, zl[-1] * zxoverL)

    polLengthArray = np.array(returnll(R, Z))
    Bx = np.abs(TotalField[Xpoint])

    # cut kinked data

    if returnSBool:
        S = returnS(R, Z, TotalField, Bpol)
        return zl, TotalField, Xpoint, R, Z, Rs, Zs, polLengthArray, Bpol, S

    return zl, TotalField, Xpoint, R, Z, Rs, Zs, polLengthArray, Bpol


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
