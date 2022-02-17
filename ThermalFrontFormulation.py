import scipy as sp
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, trapz
from scipy import interpolate
from AnalyticCoolingCurves import LfuncN
from unpackConfigurations import unpackConfiguration

def CfInt(spar, B_field, sx, L,sh = 0,kappa1=2500):
    """function which returns the control parameter required for a detachment front at parallel position sh"""
    B_field = interpolate.interp1d(spar,B_field,kind='cubic',fill_value="extrapolate")
    #calculate Tu/qpll**(2/7) by integrating over heat flux density
    Tu = quad(integrand,sh,sx,args = (sx, L, B_field),epsabs = 0.0000000000000000001)[0]
    if sx<L:
        Tu = Tu+quad(integrand2,sx,L,args = (sx, L, B_field),epsabs = 0.0000000000000000001)[0]
    Tu = (Tu*7/(2*kappa1))**(-2/7)
    #account for flux expansion effects on heat flux density
    Cf = Tu*B_field(sh)/B_field(sx)
    #account for constants related to total impurity radiation
    Q = []
    T = np.linspace(0,100,1000)
    for t in T:
        Q.append(LfuncN(t))
    C0 = (2*kappa1*trapz(Q*T**(1/2),T))**(-1/2)
    Cf = Cf*C0
    return Cf

def Tu(z, B_field, zx, L,Beta = 1, zh = 0):
    B_field = interpolate.interp1d(z,B_field,kind='cubic')
    intTop = quad(integrand,zh,L,args = (zx, L, B_field), epsabs = 0.0000000000000000001)[0]
    answer = (intTop)**(2/7)
    return answer

def averageB(s, B_field, zx, L,Beta = 1, zh = 0):
    Bfield1 = interpolate.interp1d(s,np.add(np.multiply(B_field,0),1),kind='cubic')
    B_field = interpolate.interp1d(s,np.sqrt(B_field),kind='cubic')
    int0= quad(integrand,zh,L,args = (zx, L, B_field), epsabs = 0.0000000000000000001)[0]
    int1 = quad(integrand,zh,L,args = (zx, L, Bfield1), epsabs = 0.0000000000000000001)[0]
    answer = int0/int1
    return answer


def integrand(s,sx, L, B_field):
        return (B_field(s)/B_field(sx))

def integrand2(s,sx, L, B_field):
        return  (L-s)*(B_field(s)/B_field(sx))/(L-sx)
    


