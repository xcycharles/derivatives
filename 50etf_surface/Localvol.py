import pandas as pd
import datetime
import numpy as np
from scipy import interpolate
from math import *

xls_file = pd.ExcelFile('LVParam.xlsx')

'''The most important element in this script is the LocalVol class， the important
focus in this class is the function "localvolsurface", which allow us to input a stock
price and time t, and output a local vol as an output.'''


def getprice(target):  # current spot price of the underlying
    if target == '510050':
        return 3.5710


def localvarianceline(args): # will input calibrated parameters in here
    sigatm, delta, kappa, gamma = args
    # x is log moneyness
    func = lambda x: sigatm ** 2 + delta * np.tanh(kappa * x) / kappa + \
                     gamma / 2 * (np.tanh(kappa * x) / kappa) ** 2
    return func


class LocalVol:
    def __init__(self, Target):
        self.Target = Target
        self.Para = xls_file.parse(self.Target).drop('Unnamed: 0', axis=1)
        self.column = self.Para.columns.values # time, sigatm, delta, kappa, gamma
        self.TTM = self.Para['time']

    def timeindex(self, t):
        for i in range(len(self.TTM)):
            if t < self.TTM[i]:
                return i
        return len(self.TTM) # if input TTM is too big

    def localvolsur(self, St:int or float, t):
        index = self.timeindex(t)
        if index == 0:
            return localvarianceline(self.Para.loc[index][1:])(log(St / getprice(self.Target))) ** 0.5
        elif index == len(self.TTM):
            return localvarianceline(self.Para.loc[index - 1][1:])(log(St / getprice(self.Target))) ** 0.5
        elif t in list(self.TTM):
            return localvarianceline(self.Para.loc[index - 1][1:])(log(St / getprice(self.Target))) ** 0.5
        else: # input TTM not in exactly maturity
            tt = self.Para.loc[index - 1][0]
            TT = self.Para.loc[index][0]
            # because unit variance from tt to TT is the same, so we can infer any t in between
            # (VTT-Vtt)/(TT-tt) = (Vt-Vtt)/(t-tt) = (VT-Vt)/(TT-t)
            vartt = localvarianceline(self.Para.loc[index - 1][1:])(log(St / getprice(self.Target)))
            varTT = localvarianceline(self.Para.loc[index][1:])(log(St / getprice(self.Target)))
            vart = vartt + (t - tt) * (varTT - vartt) / (TT - tt)
            return vart ** 0.5

print(f"test: {LocalVol('510050').localvolsur(3,0.5)}")



'''            sig1tt = self.Para.loc[index - 1][0] * localvolline(
                self.Para.loc[index - 1][1:])(log(St / getprice(self.Target)))
            sig2tT = self.Para.loc[index][0] * localvolline(self.Para.loc[index][1:])(log(St / getprice(self.Target)))
            sigtT = (sig2tT - sig1tt) / (self.Para.loc[index][0] - self.Para.loc[index - 1][0])'''
