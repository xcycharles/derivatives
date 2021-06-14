from datetime import datetime
import matplotlib.pyplot as plt
from functools import partial
from scipy import optimize
from mpl_toolkits.mplot3d import Axes3D
import Localvol
from matplotlib import cm
from math import *
from numpy import *
import numpy as np
import LVOptimizeCurve
import LVinitProcess
from scipy import stats
import time
start_time = time.time()


def BlackScholes(v, X, T, CallPutFlag='c', S=3.5710, r=0.015):

    d1 = (np.log(S / X) + (r + v * v / 2.) * T) / (v * np.sqrt(T))
    d2 = d1 - v * np.sqrt(T)
    if CallPutFlag == 'c':
        return S * float(stats.norm.cdf(d1)) - X * np.exp(-r * T) * float(stats.norm.cdf(d2))
    else:
        return X * np.exp(-r * T) * float(stats.norm.cdf(-d2)) - S * float(stats.norm.cdf(-d1))



# it needs to be a function of both strike and maturity
def calc_impl_vol(strike, time, right='c', underlying=3.5710, rf=0.015):
    # black scholes price - market option price
    fun = lambda x: BlackScholes(x, CallPutFlag=right, S=underlying, X=strike, T=time, r=rf) - LVinitProcess.call_payoff(strike)
    return optimize.bisect(fun, -1, 2)

Yaxis = np.linspace(0.5, 2.0, 100)  # 400 steps for maturity/365
Xaxis = np.linspace(2.5, 5.0, 50)  # 100 strikes is just right for graph showing

fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y, Z = [], [], []
for i in Xaxis: # stack along strikes
    for j in Yaxis:
        X.append(i) # Strike
        Y.append(j) # TTM
        Z.append(calc_impl_vol(strike=i,time=j))

print(len(X))
print(len(Y))
print(len(Z))

# Normalize the colors based on Z value
norm = plt.Normalize(np.min(Z), np.max(Z))
colors = cm.jet(norm(Z))


plt.title('Implied Vol Surface for 510050 call')
plt.xlabel('Strike')
plt.ylabel('Maturity')
ax.scatter(X, Y, Z, facecolors=colors, s=0.2).set_facecolor((0,0,0,0))
plt.show()

print("time elapsed: {:.2f}m".format((time.time() - start_time)/60))