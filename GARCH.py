import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as spop
import tushare as ts
from icecream import ic
import datetime as dt
from tushare_data import df
from scipy.stats.mstats import winsorize


# get future price
future = pd.read_csv('IH2103_2021-03-12.csv', index_col=0)
future.index = future.index.map(lambda x:dt.datetime.strptime(x,'%y-%m-%d %H:%M:%S.%f '))
df = future
df['LastPrice'] = df['LastPrice'].apply(lambda x: winsorize(x,limits=[0.2,0.2]))

df = df.dropna(axis=1)
training = 0.8
#prices = df['600009.SH'][:int(len(df) * training)]
prices = df['LastPrice'][:int(len(df) * training)]
returns = np.array(prices)[1:]/np.array(prices)[:-1] - 1
mean = np.average(returns)
var = np.std(returns)**2

def GARCH_MLE(params):
    #specifying model parameters
    mu = params[0] # mean
    omega = params[1] # var
    alpha = params[2] # 0 AR ARCH
    beta = params[3] # 0 ARMA GARCH
    beta2 = params[7]
    delta = params[4] # 0 risk premium
    theta = params[5] # 0 TGARCH
    kappa = params[6] # power
    #calculating long-run volatility
    long_run = (omega/(1 - alpha - beta - beta2))**(1/2) # daily

    resid = np.zeros(len(returns)) # daily realized vol
    resid[0] = returns[0]+delta*long_run-mu
    realised = abs(resid)
    conditional = np.zeros(len(returns))
    conditional[0] = long_run

    for t in range(1,len(returns)):
        resid[t] = returns[t]+delta*conditional[t-1]**2-mu
        realised[t] = abs(resid[t])
        conditional[t] = (omega+alpha*(realised[t-1]+theta*resid[t-1])**kappa+beta*conditional[t-1]**kappa+beta2*conditional[t-2]**kappa)**(1/kappa)

    #calculating log-likelihood
    likelihood = 1/((2*np.pi)**(1/2)*conditional)*np.exp(-realised**2/(2*conditional**2))
    log_likelihood = np.sum(np.log(likelihood))
    return -log_likelihood

################## main ##########################
#maximising log-likelihood
params = [mean, var, 0, 0, 0, 0, 0, var]
# minalpha = 0
# minbeta = 0
# maxalpha = 1
# maxbeta = 1
# maxalphabeta = 1
# minmu = -0.001
# maxmu = 0.001
# minvar = 0.00000000001
# maxvar = var
# cons = [
#         {'type':'ineq','fun':lambda x:x[2]-minalpha},
#         {'type':'ineq','fun':lambda x:maxalpha-x[2]},
#         {'type':'ineq','fun':lambda x:x[3]-minbeta},
#         {'type':'ineq','fun':lambda x:maxbeta-x[3]},
#         {'type':'ineq','fun':lambda x:maxalphabeta-(x[2]+x[3])},
#         {'type':'ineq','fun':lambda x:maxmu-x[0]},
#         {'type':'ineq','fun':lambda x:x[0]-minmu},
#         {'type':'ineq','fun':lambda x:maxvar-x[1]},
#         {'type':'ineq','fun':lambda x:x[1]-minvar},
#         ]
bnds = ((-0.001,0.001),(1e-9,var),(0,1),(0,1),(0,1),(0,1),(0,5),((1e-9,var))) # note result should not be very close to the bound
#res = spop.minimize(GARCH_MLE, x0=np.array(params), method='SLSQP', bounds=bnds)
#res = spop.minimize(GARCH_MLE, x0=np.array(params), method='COBYLA', bounds=bnds)
#res = spop.minimize(GARCH_MLE, np.asarray(params), method='Nelder-Mead')
res = spop.minimize(GARCH_MLE, np.asarray(params), method='Powell', bounds=bnds)
#res = spop.minimize(GARCH_MLE, np.asarray(params), method='BFGS', bounds=bnds)
mu = res.x[0]
omega = res.x[1]
alpha = res.x[2]
beta = res.x[3]
delta = res.x[4]
theta = res.x[5]
kappa = res.x[6]
beta2 = res.x[7]
log_likelihood = -float(res.fun)
long_run = (omega/(1 - alpha - beta - beta2))**(1/2)
print(f'GARCH model parameters are:')
print('mu '+str(round(mu, 6)))
print('omega '+str(round(omega, 6)))
print('alpha '+str(round(alpha, 4)))
print('beta '+str(round(beta, 4)))
print('beta2 '+str(round(beta2,4)))
print('delta '+str(round(delta,4)))
print('theta '+str(round(theta,4)))
print('kappa '+str(round(kappa,4)))
print('log-likelihood '+str(round(log_likelihood, 4)))
print('long-run volatility '+str(round(long_run, 4)))

#calculating realised and conditional volatility for optimal parameters
#prices = df['600009.SH']
prices = df['LastPrice']
returns = np.array(prices)[1:]/np.array(prices)[:-1] - 1
#mean = np.average(returns) # assume constant mean from the training set
var = np.std(returns)**2
long_run = (omega/(1 - alpha - beta - beta2))**(1/2)

resid = np.zeros(len(returns))  # daily realized vol
resid[0] = returns[0] + delta * long_run - mu
realised = abs(resid)
conditional = np.zeros(len(df)-1) # minus first day retur
conditional[0] = long_run

for t in range(1, len(df)):
    try:
        resid[t] = returns[t] + delta * conditional[t - 1] ** 2 - mu
        realised[t] = abs(resid[t])
        conditional[t] = (omega + alpha * (realised[t - 1] + theta * resid[t - 1]) ** kappa + beta * conditional[t - 1] ** kappa+ beta2 * conditional[t - 2] ** kappa) ** (1 / kappa)
    except:
        pass

#visualising the results
plt.figure(figsize=(16,10))
plt.rc('xtick', labelsize = 8)
# from 1 day due to return calculation
plt.plot(df.index[1:][::50],realised[::50],label='realised',linewidth=0.5)
plt.plot(df.index[1:][::50],conditional[::50],label='conditional',linewidth=0.5)
plt.legend()
plt.grid()
plt.axvline(x=df.iloc[int(len(df) * training)].name,color='red')
plt.savefig('aa.png')
plt.show()

plt.figure()
plt.scatter(conditional,realised)
plt.show()

print(np.corrcoef(conditional[int(len(df) * training):],realised[int(len(df) * training):]))