import numpy as np
from scipy.integrate import quad, trapezoid
from scipy.interpolate import interp1d

from .AnalyticCoolingCurves import LfuncN


def CfInt(spar, B_field, sx, L, sh=0, kappa1=2500):
    """Calculate the control parameter required for a detachment front
    at parallel position ``sh``

    Parameters
    ----------
    spar: array, m
        Array of S parallel
    B_field: array, T
        Array of total B field
    sx: float, m
        Position of detachment front in the parallel
    L: float, m
        Connection length in the parallel
    sh: float, m
        Parallel front position
    kappa1: float, W/m^2/K^7/2
        Electron thermal conductivity

    """

    B_field = interp1d(spar, B_field, kind="cubic", fill_value="extrapolate")
    # calculate Tu/qpll**(2/7) by integrating over heat flux density
    epsabs = 1e-18
    Tu = quad(_integrand, sh, sx, args=(sx, B_field), epsabs=epsabs)[0]
    if sx < L:
        Tu += quad(_integrand2, sx, L, args=(sx, L, B_field), epsabs=epsabs)[0]
    Tu = (Tu * 7 / (2 * kappa1)) ** (-2 / 7)
    # account for flux expansion effects on heat flux density
    Cf = Tu * B_field(sh) / B_field(sx)
    # account for constants related to total impurity radiation
    T = np.linspace(0, 100, 1000)
    Q = np.array([LfuncN(t) for t in T])
    C0 = 1.0 / np.sqrt(2 * kappa1 * trapezoid(Q * np.sqrt(T), T))
    return Cf * C0


def _integrand(s, sx, B_field):
    return B_field(s) / B_field(sx)


def _integrand2(s, sx, L, B_field):
    return (L - s) * (B_field(s) / B_field(sx)) / (L - sx)
