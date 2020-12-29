# %%
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad,trapz, cumtrapz, odeint, solve_ivp
from scipy import interpolate
import ThermalFrontFormulation as TF
from unpackConfigurations import unpackConfiguration,returnzl,returnll
from matplotlib.collections import LineCollection

ionisAdas = np.loadtxt("ionisation.dat")
tempionis = ionisAdas[0]
ionisFunc = interpolate.interp1d(tempionis,ionisAdas[1]/(10**6),fill_value='extrapolate')

#Nitrogen based cooling curve used in Lipschultz 2016
def Lfunc(T):
   answer = 0
   if T >= 1 and T<= 80:
       answer = 5.9E-34*(T-1)**(0.5)
       answer = answer*(80-T)
       answer = answer/(1+(3.1E-3)*(T-1)**2)
   else:
       answer = 0
   return answer

#Ne based cooling curve produced by Matlab polynominal curve fitting "polyval" (Ryoko 2020 Nov)
#def Lfunc(T):    
#    answer = 0
#    if T >= 3 and T<= 100:
#    	answer = -2.0385E-40*T**5 + 5.4824E-38*T**4 -5.1190E-36*T**3 + 1.7347E-34*T**2 -3.4151E-34*T -3.2798E-34
#    elif T >=2 and T < 3:
#       answer = (8.0-1.0)*1.0E-35/(3.0-2.0)*(T-2.0)+1.0E-35
#    elif T >=1 and T < 2:
#       answer = 1.0E-35/(2.0-1.0)*(T-1.0)
#    else:
#        answer = 0
#    return answer

#plt.plot(T,Lfunc)
#plt.show()

#Ar based cooling curve produced by Matlab polynominal curve fitting "polyval" (Ryoko 2020 Nov)
# def Lfunc(T):
#     answer = 0
#     if T >= 1.5 and T<= 100:
#         answer = -4.9692e-48*T**10 + 2.8025e-45*T**9 -6.7148e-43*T**8 + 8.8636e-41*T**7 -6.9642e-39*T**6 +3.2559e-37*T**5 -8.3410e-36*T**4 +8.6011e-35*T**3 +1.9958e-34*T**2 + 4.9864e-34*T -9.9412e-34
#     elif T >= 1.0 and T< 1.5:
#         answer = 2.5E-35/(1.5-1.0)*(T-1.0)
#     else:
#         answer = 0
#     return answer

#Custom gaussian impurity cooling curve if desired
# def LfunLengFunccGauss(T,width = 10):
#     return 1E-31*np.exp(-(T-10)**2/(width))

#Function to integrate, that returns dq/ds and dt/ds using Lengyel formulation and field line conduction
def LengFunc(y,s,kappa0,nu,Tu,cz,qpllu0,radios,S):
    qoverB,T = y
    fieldValue = 0
    # in the case that the RK x values are outside our interpolated field range, set B equal to the limits of B
    if s > S[-1]:
        fieldValue = B(S[-1])
    elif s< S[0]:
        fieldValue = B(S[0])
    else:
        fieldValue = B(s)
    #add a constant radial source of heat above the X point
    if radios["upstreamGrid"]:
        if s >S[Xpoint]:
            dqoverBds = ((nu**2*Tu**2)/T**2)*cz*Lfunc(T) - qpllu0/np.abs(S[-1]-S[Xpoint])
        else:
            dqoverBds = ((nu**2*Tu**2)/T**2)*cz*Lfunc(T) 
    else:
        dqoverBds = ((nu**2*Tu**2)/T**2)*cz*Lfunc(T) 
    # working on neutral/ionisation model
    # if radios["ionisation"] == True:
    #     dqoverBds = dqoverBds + ((nu**2*Tu**2)/T**2)*ionisFunc(T)*13*1.60*10**(-19)
    dqoverBds = dqoverBds/fieldValue
    dtds = qoverB*fieldValue/(kappa0*T**(5/2))
    if qoverB < 0:
        dtds = 0
    #return gradient of q and T
    return [dqoverBds,dtds]

def returnImpurityFracLeng(constants,radios,S,indexRange,dispBassum = False,dispqassum = False,dispUassum = False):
    C = []
    Tus = []
    splot = []
    #lay out constants
    gamma_sheath = constants["gamma_sheath"]
    qpllu0 = constants["qpllu0"]
    nu = constants["nu"]
    kappa0 = constants["kappa0"]
    mi = constants["mi"]
    echarge = constants["echarge"]
    Tt = constants["Tt"]

    for i in indexrange:
        print(i)
        #current set of parallel position coordinates
        s = S[i:-1]
        splot.append(S[i])
        error0 = 1

        #create the nitrogen based cooling curve
        T = np.linspace(1,100,100)#should be this for Ar? Ryoko 20201209 --> almost no effect
        Lalpha = []
        for DT in T:
            Lalpha.append(Lfunc(DT))
        Lalpha = np.array(Lalpha)
        T = np.append(T,500)
        Lalpha = np.append(Lalpha,0)
        T = np.append(1,T)
        Lalpha = np.append(0,Lalpha)
        T = np.append(0,T)
        Lalpha = np.append(0,Lalpha)
        Lz = [T,Lalpha]

        #inital guess for the value of qpll integrated across connection length
        qavLguess = 0
        if radios["upstreamGrid"]:
            if s[0] < S[Xpoint]:
                qavLguess = ((qpllu0)*(S[Xpoint]-s[0]) + (qpllu0/2)*(s[-1]-S[Xpoint]))/(s[-1]-S[0])
                #qavLguess = ((qpllu0)*(S[Xpoint]-s[0]) + (qpllu0)*(s[-1]-S[Xpoint]))/(s[-1]-S[0]) #for consistency when choosing the option (a)?
            else:
                qavLguess = (qpllu0/2)
        else:
            qavLguess = (qpllu0)
        #inital guess for upstream temperature based on guess of qpll ds integral
        Tu0 = ((7/2)*qavLguess*(s[-1]-s[0])/kappa0)**(2/7)
        Tu = Tu0
        #iterate through temperature untill the consistent Tu is determined
        while np.abs(error0) > 0.0005:
            print("Tu="+str(Tu))
            Lint = cumtrapz(Lz[1]*np.sqrt(Lz[0]),Lz[0],initial = 0)
            integralinterp = interpolate.interp1d(Lz[0],Lint)
            #initial guess of cz0 assuming qpll0 everywhere and qpll=0 at target
            cz0 = (qpllu0**2 )/(2*kappa0*nu**2*Tu**2*integralinterp(Tu))
            cz = cz0
            #set initial percentage change of cz after every guess
            perChange = 0.5
            error1 = 1
            switch0 = 0
            swtich1 = 0
            swapped = 0
            #iterate until the correct impurity fraction is found
            while perChange >0.0005:
                #initial guess of qpllt, typically 0
                qpllt = gamma_sheath/2*nu*Tu*echarge*np.sqrt(2*Tt*echarge/mi)
                result = odeint(LengFunc,y0=[qpllt/B(s[0]),Tt],t=s,args=(kappa0,nu,Tu,cz,qpllu0,radios,S))
                q = result[:,0]*B(s)
                T = result[:,1]
                qpllu1 = q[-1]
                if radios["upstreamGrid"]:
                    error1 = (qpllu1-0)/qpllu0
                else:
                    error1 = (qpllu1-qpllu0)/qpllu0
                if 0<error1:
                    switch1 = 0
                    #increase counter 'swapped' if the guess overshoots
                    if switch1!=switch0:
                        swapped += 1
                    else:
                        swapped = 0
                    cz = cz*(1-perChange)
                else:
                    switch1 = 1
                    if switch1!=switch0:
                        swapped += 1
                    else:
                        swapped = 0
                    cz = cz*(1+perChange)
                switch0 = switch1
                #if the guess has overshot twice, decrease the change in cz per guess
                if swapped >1:
                    perChange = perChange*0.1
                    swapped = 0
            Tucalc = T[-1]
            Tu = 0.8*Tu + 0.2*Tucalc
            error0 = (Tu-Tucalc)/Tu
        Q = []
        for Tf in T:
            Q.append(Lfunc(Tf))
        if dispBassum == True:
            #evaluate the accuracy of the assumption that B is constant through the radiating region
            BassumpAccuracy = trapz(Q*T**0.5,T)/trapz(Q*T**0.5/B(s),T)
            BassumpAccuracy = np.abs(1-BassumpAccuracy/B(S[-1]))*100
            print("assumption of constant field is accurate to "+str(BassumpAccuracy)+"%")
        if dispqassum == True:
            #evaluate the accuracy of the assumption that q is constant up until the radiating region
            QassumAccuracy = (trapz(q/kappa0,s))**(2/7)
            QassumAccuracy = QassumAccuracy/(trapz(np.add(np.multiply(q,0),qpllu0)/kappa0,s))**(2/7)
            QassumAccuracy = np.abs(1-QassumAccuracy)*100
            print("assumption of constant qpll is accurate to "+str(QassumAccuracy)+"%")
        if dispUassum == True:
            #evaluate the accuracy of the assumption that q is constant up until the radiating region
            UassumAccuracy = np.sqrt(trapz(Q*T**0.5,T))
            print("the current value of U is "+str(UassumAccuracy))
        Tus.append(Tu)
        C.append(np.sqrt(cz))
    return splot, C


# %%
gridFile = "C:\\Users\\cydco\\Desktop\\My_stuff\\PhD\\Year 1\\Understanding Super x field\\balance.nc"
zl, TotalField, Xpoint,R0,Z0,R,Z, Spol, Bpol, S = unpackConfiguration(File = gridFile,
    Type ="outer",returnSBool = True,sepadd=2)

plt.plot(np.transpose(R0),np.transpose(Z0),color="C3",label="SOL ring chosen")
plt.axes().set_aspect('equal')
segs1 = np.stack((R,Z), axis=2)
segs2 = segs1.transpose(1,0,2)
plt.xlim([np.amin(R),np.amax(R)])
plt.ylim([np.amin(Z),0])

plt.gca().add_collection(LineCollection(segs1))
plt.gca().add_collection(LineCollection(segs2))
plt.legend()
plt.show()

B =  interpolate.interp1d(S, TotalField, kind='cubic')
plt.plot(S,TotalField)
plt.xlabel("s (m)")
plt.ylabel("B (T)")
plt.savefig("field.png", dpi = 400)
plt.show()


#define the range along the field line we want to calculate C for
indexrange = np.linspace(0,Xpoint-10,7)
indexrange = list(indexrange.astype(int))

# unpack field line data in parallel coordinate z (used for thermal front model)
L = zl[-1]
zx = zl[Xpoint]
zhoverLrange= zl[indexrange]/zl[-1]

Bpol = np.sqrt(Bpol**2)
LS = S[-1]
# generate detachment parameter data from the simple thermal front model
CoverCxTF = []
TusTF = []
for j in zhoverLrange:
            CoverCxTF.append(1/TF.CXoverChInt(zl, TotalField, zx, L,Beta = 1,zh = j*L))
            TusTF.append(TF.Tu(zl, TotalField, zx, L,Beta = 1,zh = j*L))


# %%
constants = {
    "gamma_sheath": 7,
    "qpllu0": 7*10**8,
    "nu" : 1*10**20,
    "kappa0" : 1.5*10**4,
    "mi": 3*10**(-27),
    "echarge": 1.60*10**(-19),
    "Tt": 0.2,
}

radios = {
    "ionisation": False,
    "upstreamGrid": True, #if true, source of divertor heat flux comes from radial transport upstream, and Tu is at the midplane. If false, heat flux simply enters at the x point as qi, and Tu is located at the x point. 
}


splot,C = returnImpurityFracLeng(constants,radios,S=S,indexRange=indexrange)
radios["upstreamGrid"] = False
splot,C2 = returnImpurityFracLeng(constants,radios,S=S[:Xpoint],indexRange=indexrange)
# %%
Spolplot  = Spol[indexrange]/Spol[-1]
# plt.plot(srange/LS,np.divide(np.array(Tus),Tus[0]),label="thermal front")
# returnImpurityFracLeng(gamma_sheath,qpllu0,Tt,nu,kappa0,mi,echarge,dispBassum=False,dispqassum=False,dispUassum=False,neutralmodel=False)

plt.plot(Spolplot,np.divide(np.array(CoverCxTF),CoverCxTF[0]),label="thermal front")
plt.plot(Spolplot,C/C[0],label="lengyel")
plt.plot(Spolplot,C2/C2[0],label="lengyel no upstream")
plt.xlabel("spol/Lpol")
plt.ylabel("C/CX")
# plt.ylabel("Tu (eV)")
plt.legend()
plt.savefig("ControlParameter.png",dpi=400)
plt.show()

# %%
plt.plot(Spolplot,np.divide(np.gradient(CoverCxTF),
    np.gradient(Spolplot)*CoverCxTF),label="thermal front")
plt.plot(Spolplot,np.gradient(C)/(np.gradient(Spolplot)*C),label="lengyel + ionisation")

plt.xlabel("spol/Lpol")
plt.ylabel("sensitivity")
# plt.ylabel("Tu (eV)")
plt.legend()
plt.savefig("sensitivity.png",dpi=400)
plt.show()
# %%
