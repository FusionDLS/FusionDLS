
# %%
from LengyelReinkeFormulation import LengFunc,returnImpurityFracLeng
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad,trapz, cumtrapz, odeint, solve_ivp
from scipy import interpolate
import ThermalFrontFormulation as TF
from unpackConfigurations import unpackConfiguration,returnzl,returnll
from matplotlib.collections import LineCollection
from AnalyticCoolingCurves import LfuncN,LfunLengFunccGauss,LfuncNe
# %%

#import geometry
gridFile = "testGrids\\ConnectionLength\\L1.00.nc"
zl,TotalField,Xpoint,R0,Z0,R,Z, polLengthArray, Bpol,S = unpackConfiguration(gridFile,"Box",zxoverL = 0.7, returnSBool = True,polModulator = 1,sepadd = 0)

#set switches
radios = {
    "ionisation": False,  # in development
    "upstreamGrid": False, #if true, source of divertor heat flux comes from radial transport upstream, and Tu is at the midplane. If false, heat flux simply enters at the x point as qi, and Tu is located at the x point. 
    "fluxlim": False,  # if true, turns on a flux limiter with coefficient alpha
}

#set general run parameters
constants = {
    "gamma_sheath": 7, #sheath transmittion coefficient for virtual target. Choice does not matter if Tt is low
    "qpllu0": 5*10**9, # heat flux density at the x point
    "nu" : 1*10**19, #upstream density
    "kappa0" : 2500,
    "mi": 3*10**(-27),
    "echarge": 1.60*10**(-19), 
    "Tt": 0.5, # requested virtual target temp. Aim for low (sub 1eV) Tt
    "Lfunc": LfuncN, #impurity cooling function
    "alpha": 1000, #flux limiting alpha. Only matters if fluxlim is true
}

# %%

#range of indices to choose for front position
indexrange = np.linspace(0,Xpoint-10,4)
indexrange = list(indexrange.astype(int))

#calculate C using simpler thermal front model
Sf = []
Cf = []
for i in indexrange:
    Sf.append(S[i])
    Cf.append(TF.CfInt(S, TotalField, S[-1], S[-1],sh = S[i],kappa1=2500))


# %%
B =  interpolate.interp1d(S, TotalField, kind='cubic')

constants["XpointIndex"] = Xpoint
constants["B"] = interpolate.interp1d(S, TotalField, kind='cubic')

#calculate C using the Lengyel-Reinke model
splot,C,Sprofiles,Tprofiles,Qprofiles = returnImpurityFracLeng(constants,radios,S,indexrange)
# %%

# compare models against eachother
plt.plot(Sf,Cf/Cf[0],label="thermal front",color="C0",marker="o")
plt.plot(splot,C/C[0],label="lengyel-Reinke",color="C1")
plt.ylabel("spol/Lpol")
plt.xlabel("C/CX")
plt.legend()
plt.savefig("ControlParameter.png",dpi=400)
plt.show()

#plot temperature profiles from Lengyel-Reinke model
for i in range(len(Tprofiles)):
    plt.plot(Sprofiles[i],Tprofiles[i])
plt.ylabel("Te (eV) (m)")
plt.xlabel("s|| (m)")
plt.show()

#plot heat flux profiles from Lengyel-Reinke model
for i in range(len(Qprofiles)):
    plt.plot(Sprofiles[i],Qprofiles[i])
plt.ylabel("qpll")
plt.xlabel("s|| (m)")
plt.show()
# %%
