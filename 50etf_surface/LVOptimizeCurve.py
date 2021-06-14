import time

import pandas as pd
import datetime
import numpy as np
from scipy import interpolate
from scipy.optimize import minimize
import Yield as yd
from scipy.misc import derivative
import matplotlib.pyplot as plt

'''This script is basically used to fit a curve that are suitable for the 
Local vol point for each point t. Here we used 'SLSPQ' from 
scipy.optimize.minimize to do this curve fitting. '''


def getprice(target):
    if target == '510050':
        return 3.5710


Indexlist = ['510050']
#lv_xls_file = pd.ExcelFile('Local volatility.xlsx')
lv_xls_file = pd.ExcelFile('local vol raw.xlsx')
# this is atm implied vol to fix the vol surface on
iv_xls_file = pd.ExcelFile('impliedvol.xlsx')
writer = pd.ExcelWriter('LVParam.xlsx')

target: str
for target in Indexlist:
    try:
        LV = lv_xls_file.parse(target)
        LV = LV.drop('Unnamed: 0', axis=1)
        IV = iv_xls_file.parse(target)

        x = list(IV['Term']) # TTM
        y = list(IV['100%']) # implied vol
        IV_cubicspline = interpolate.CubicSpline(x, y)

        column = LV.columns.values
        LV[column[0]] = LV[column[0]].div(getprice(target), axis='index').apply(np.log) # convert to log moneyness
        Time, Sigatm, Delta, Kappa, Gamma = [], [], [], [], []

        for i in range(len(column) - 1): # for all the expiries
            i += 1
            data = LV[[column[0], column[i]]] # one strike one expiry of local vol
            data = data[data[column[i]] > 0] # check for positive local vol
            X = list(data[column[0]]) # the remaining strikes
            Y = list(data[column[i]].apply(np.square)) # all the local variance


            def targetfun(args):
                delta, kappa, gamma = args
                # daily sigatm is fixed from implied vol from bbg
                func = lambda x: IV_cubicspline(float(column[i]) / 365) ** 2 \
                                 + delta * np.tanh(kappa * x) / kappa + \
                                 gamma / 2 * (np.tanh(kappa * x) / kappa) ** 2
                # x here is log moneyness
                # IV_cubicspline inputs TTM and outputs implied vol
                error = 0
                for j in range(len(X)): # for all the strikes
                    error += (func(X[j]) - Y[j]) ** 2
                return error


            def con(args):
                dmin, dmax, kmin, kmax, gmin, gmax = args
                # sigatm is fixed
                # delta x[0] is neg and controls skewness
                # kappa x[1] controls the wings
                # gamma x[2] is the convexity
                cons = (
                        {'type': 'ineq', 'fun': lambda x: x[0] - dmin},
                        {'type': 'ineq', 'fun': lambda x: dmax - x[0]},
                        {'type': 'ineq', 'fun': lambda x: x[1] - kmin},
                        {'type': 'ineq', 'fun': lambda x: kmax - x[1]},
                        {'type': 'ineq', 'fun': lambda x: x[2] - gmin},
                        {'type': 'ineq', 'fun': lambda x: gmax - x[2]}
                )
                # ineq means constraint function result is non-negative
                return cons


            dmin, dmax, kmin, kmax, gmin, gmax = -0.05, 0.0, 2, 3.5, 0.35, 1
            if len(X) < 2: # only 2 strikes with positive local vol
                dmin, dmax, kmin, kmax, gmin, gmax = -0.05, 0.0, 1, 2, 0.03, 0.05
            elif column[i] / 365 > 1.2: # long maturities
                dmin, dmax, kmin, kmax, gmin, gmax = -0.05, 0.0, 1, 2, 0.03, 0.1
            args1 = (dmin, dmax, kmin, kmax, gmin, gmax)
            cons = con(args1)
            guess = np.asarray((1, 1, 1)) # guess for delta kappa gamma
            res = minimize(targetfun, guess, method='SLSQP', constraints=cons)
            print(res.success)
            #print(res.x)

            Xaxis = np.linspace(-4, 4, 100) # log moneyness
            delta, kappa, gamma = res.x
            localvar = list(map(lambda x: IV_cubicspline(float(column[i]) / 365) ** 2 + delta * np.tanh(kappa * x) / kappa + \
                                          gamma / 2 * (np.tanh(kappa * x) / kappa) ** 2, Xaxis))
            Yaxis = list(map(lambda x: x ** 0.5, localvar)) # local vol
            plt.plot(Xaxis, Yaxis)
            plt.title(column[i] / 365)
            plt.xlabel('K/S')
            # plt.show() # shows vol curve for each expiry
            sigatm = IV_cubicspline(float(column[i]) / 365)
            Time.append(float(column[i]) / 365) # TTM
            Sigatm.append(sigatm)
            Delta.append(delta)
            Kappa.append(kappa)
            Gamma.append(gamma)

        output=pd.DataFrame({'time':Time,'sigatm':Sigatm,'delta':Delta,'kappa':Kappa,'gamma':Gamma})
        output.to_excel(writer, target)
        writer.save()
    except:
        pass



