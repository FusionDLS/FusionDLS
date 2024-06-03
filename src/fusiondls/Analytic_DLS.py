import numpy as np
from scipy import interpolate
from scipy.integrate import quad, trapz

from .AnalyticCoolingCurves import LfuncN


def CfInt(spar, B_field, sx, L, sh=0, kappa1=2500):
    """
    spar: array, m
        Array of S parallel
    B_field: array, T
        Array of total B field
    sx: float, m
        Position of detachment front in the parallel
    L: float, m
        Connection length in  the parallel
    sh: float, m
        Parallel front position
    kappa1: float, W/m^2/K^7/2
        Electron thermal conductivity

    """
    """function which returns the control parameter required for a detachment front at parallel position sh"""
    B_field = interpolate.interp1d(
        spar, B_field, kind="cubic", fill_value="extrapolate"
    )
    # calculate Tu/qpll**(2/7) by integrating over heat flux density
    Tu = quad(integrand, sh, sx, args=(sx, L, B_field), epsabs=0.0000000000000000001)[0]
    if sx < L:
        Tu = (
            Tu
            + quad(
                integrand2, sx, L, args=(sx, L, B_field), epsabs=0.0000000000000000001
            )[0]
        )
    Tu = (Tu * 7 / (2 * kappa1)) ** (-2 / 7)
    # account for flux expansion effects on heat flux density
    Cf = Tu * B_field(sh) / B_field(sx)
    # account for constants related to total impurity radiation
    Q = []
    T = np.linspace(0, 100, 1000)
    for t in T:
        Q.append(LfuncN(t))
    C0 = (2 * kappa1 * trapz(Q * T ** (1 / 2), T)) ** (-1 / 2)
    Cf = Cf * C0
    return Cf


def integrand(s, sx, L, B_field):
    return B_field(s) / B_field(sx)


def integrand2(s, sx, L, B_field):
    return (L - s) * (B_field(s) / B_field(sx)) / (L - sx)
