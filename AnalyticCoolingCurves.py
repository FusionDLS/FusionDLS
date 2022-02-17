
import numpy as np
from scipy import interpolate
from numpy import vectorize
import pandas as pd
import matplotlib.pyplot as plt

#Nitrogen based cooling curve used in Lipschultz 2016
def LfuncN(T):
   answer = 0
   if T >= 1 and T<= 80:
       answer = 5.9E-34*(T-1)**(0.5)
       answer = answer*(80-T)
       answer = answer/(1+(3.1E-3)*(T-1)**2)
   else:
       answer = 0
   return answer


#Ne based cooling curve produced by Matlab polynominal curve fitting "polyval" (Ryoko 2020 Nov)
def LfuncNe(T):    
    answer = 0
    if T >= 3 and T<= 100:
        answer = -2.0385E-40*T**5 + 5.4824E-38*T**4 -5.1190E-36*T**3 + 1.7347E-34*T**2 -3.4151E-34*T -3.2798E-34
    elif T >=2 and T < 3:
        answer = (8.0-1.0)*1.0E-35/(3.0-2.0)*(T-2.0)+1.0E-35
    elif T >=1 and T < 2:
        answer = 1.0E-35/(2.0-1.0)*(T-1.0)
    else:
        answer = 0
    return answer

#Ar based cooling curve produced by Matlab polynominal curve fitting "polyval" (Ryoko 2020 Nov)
def LfuncAr(T):
    answer = 0
    if T >= 1.5 and T<= 100:
        answer = -4.9692e-48*T**10 + 2.8025e-45*T**9 -6.7148e-43*T**8 + 8.8636e-41*T**7 -6.9642e-39*T**6 +3.2559e-37*T**5 -8.3410e-36*T**4 +8.6011e-35*T**3 +1.9958e-34*T**2 + 4.9864e-34*T -9.9412e-34
    elif T >= 1.0 and T< 1.5:
        answer = 2.5E-35/(1.5-1.0)*(T-1.0)
    else:
        answer = 0
    return answer

# #Custom gaussian impurity cooling curve if desired
def LfunLengFunccGauss(T,width = 2):
    return 1E-31*np.exp(-(T-5)**2/(width))

# reader for AMJUL files
def ratesAmjul(file,T,n):
    rawdata = np.loadtxt(file)
    unpackedData = []
    counter = 0
    rates =0
    for i in range(3):
        for j in range(3):
            section = rawdata[int(i*len(rawdata)/3):int((i+1)*len(rawdata)/3)][:,j]
            nei = np.log(n*1E-14)**(counter)
            counter = counter+1
            for ti in range(9):
                tei = np.log(T)**(ti)
                rates = rates+tei*nei*section[ti]

    rates = np.exp(rates)

    return rates*1E-6

# reader for AMJUL CX files
def ratesAmjulCX(file,T,E):
    rawdata = np.loadtxt(file)
    unpackedData = []
    counter = 0
    rates =0
    for i in range(3):
        for j in range(3):
            section = rawdata[int(i*len(rawdata)/3):int((i+1)*len(rawdata)/3)][:,j]
            nei = np.log(E)**(counter)
            counter = counter+1
            for ti in range(9):
                tei = np.log(T)**(ti)
                rates = rates+tei*nei*section[ti]

    rates = np.exp(rates)

    return rates*1E-6
