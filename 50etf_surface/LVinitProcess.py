import pandas as pd
import datetime
import numpy as np
from scipy import interpolate
import Yield
from scipy.misc import derivative
import matplotlib.pyplot as plt

'''This is the first step of data processing. Basically it used pandas to
filter the invalid datapoint from the Excel file on option price, and
 finally calculate the local volatility points for some strike and maturity'''

xls_file = pd.ExcelFile('510050_option_prices.xlsx')
AMC = xls_file.parse('Sheet1')  # option table
expires = list(AMC["Expires"].drop_duplicates())
rt = Yield.spotrate('CN')  # cubic spline object



def inst_forward_rate(t):
    '''
    we need instantaneous forward rate tao because later Gatheral need to convert to forward call price
    spot yield curve is wrt now
    if use forward yield curve, it will be hard to match option expiries
    so convert stop rate to instantaneous forward rate
    '''
    # input t is TTM in terms of year ranging from 0-3
    fd = derivative(rt, t, dx=1e-6)  # derivative of yield curve cubic spline object at t
    # rt(t) is the spot on the yield curve
    # r_t*t=\int_0^t{tao_x*d_x}
    # (r_t*t)'=\tao_t
    # \tao_t = r_t+r_t'*t
    return rt(t) + fd * t


def isnum(x):
    if type(x) == str:
        return x[0].isnumeric()  # check if exp starts with number
    else:
        return False


newexp = filter(isnum, expires)  # filter out non date like stuff
Expdate = list(newexp)  # list of expiries
#Timedelta = [1]
Timedelta = [1,1]
for i in range(len(Expdate)): # 0, 1, 2, 3
    if i < len(Expdate) - 1: # i < 3
        F = datetime.datetime.strptime(Expdate[i], '%m/%d/%y')
        G = datetime.datetime.strptime(Expdate[i + 1], '%m/%d/%y')
        H = G - F
        Timedelta.append(H.days)
#    else: # shouldn't have this for partial C partial timedelta
#        Timedelta.append(1)
Timedelta = list(map(lambda x: x / 365, Timedelta))  # differences between expiry in unit of year

TTM = [0]
for i in range(len(Expdate)):
    F = datetime.datetime.strptime('5/17/21', '%m/%d/%y')
    G = datetime.datetime.strptime(Expdate[i], '%m/%d/%y')
    H = G - F
    TTM.append(H.days)

TTM = list(map(lambda x: x / 365, TTM))  # time to maturity in unit of year

Call = []
Put = []
for i in Expdate:
    valid_expiry = AMC[AMC['Expires'] == i]
    # note there could be problem with using last, when it's previous day trades outside bid-ask
    C = valid_expiry[['Strike', 'Last']]
    P = valid_expiry[['Strike.1', 'Last.1']]
    Call.append(C)
    Put.append(P)
    # get list of expiry of dataframe of many rows by strike and price

for j in range(len(Call)):  # for each expiry
    if j == 0:  # spot month
        call_matrix = Call[0]
        put_matrix = Put[0]
    else:
        call_matrix = pd.merge(call_matrix, Call[j], how='inner', on="Strike")
        put_matrix = pd.merge(put_matrix, Put[j], how='inner', on="Strike.1")
    # get rows of each strike and all the expiry prices

call_matrix = call_matrix.sort_values(by='Strike')  # Call
put_matrix = put_matrix.sort_values(by='Strike.1')  # Put
Colname = ['Strike']
Colname.extend(Expdate)

call_matrix.columns = Colname
put_matrix.columns = Colname
K_sq_call = call_matrix['Strike'] * call_matrix['Strike']  # series
K_sq_put = put_matrix['Strike'] * put_matrix['Strike']
call_strike = call_matrix['Strike']
put_strike = put_matrix['Strike']

## must not use cublicspline to interpret option prices directly
for i in Expdate:
    call_price = call_matrix[i]
    put_price = put_matrix[i]
    call_payoff = interpolate.CubicSpline(call_strike, call_price)
    put_payoff = interpolate.CubicSpline(put_strike, put_price)
#     call_matrix[i] = call_matrix['Strike'].apply(call_payoff)
#     put_matrix[i] = put_matrix['Strike'].apply(put_payoff)

# for call
Bullspread = call_matrix.diff(-1)  # diff with next row, so sell higher strike
Strikediff = -Bullspread['Strike']  # due to increasing strike
Bullspread = Bullspread.div(Strikediff, axis='index')  # price div by strike to get moneyness
Bullspread[Bullspread <= 0] = np.nan  # surface should not have arbitrage
# Bullspread[Bullspread > 3] = np.nan  # general filter for bad option prices due to bid/ask spread
BullButterfly = Bullspread.diff(-1).div(Strikediff, axis='index')  # diff with next row, so sell higher strike bull spread
Calender = call_matrix.diff(1, axis=1)  # sell shorter maturity (diff with previous row)
Calender = Calender.fillna(np.nan)  # np.nan stores floating point, pd.nan stores integer
Calender[Calender <= 0] = np.nan  # longer maturity options are more expensive, which we buy; so under no arbitrage, calender price should>0
Calender = Calender.div(Timedelta, axis=1)  # partial C divide by partial T
BullButterfly[BullButterfly < 0] = np.nan  # buy lower strike bull spread whose price is higher (more ITM)
# denominator for spot option price = 0.5*K^2*(partial^2_C/partial_K^2)
denominator = 0.5 * BullButterfly.multiply(K_sq_call, axis='index')  # elewise multiply
denominator[denominator == 0] = np.nan
# numerator for spot option price = partial_C/partial_T+(rt-dt)(C-K(partial_C/partial_K))
CminusKBS = call_matrix.sub(Bullspread.multiply(call_matrix['Strike'], axis='index'))
RTminusDT = list(map(lambda x: inst_forward_rate(x) - Yield.div_yield_china, TTM))
Modifier = CminusKBS.multiply(RTminusDT)
Numerator = Calender.add(Modifier)
Numerator[Numerator <= 0] = np.nan  # should be strictly positive for local variance
SimpleLVar = Numerator.div(denominator)
SimpleLV = SimpleLVar.apply(lambda x: x ** 0.5)  # cannot use np.sqrt due to nan values
# SimpleLV[SimpleLV < 0.05] = np.nan
# SimpleLV[SimpleLV > 1] = np.nan
SimpleLV['Strike'] = call_matrix['Strike']


# easy for 'LVOptimizeCurve.py' processing
col = ['Strike']
col.extend(list(map(lambda x: int(x * 365), TTM[1:]))) # first TTM is zero
SimpleLV.columns = col

writer = pd.ExcelWriter('local vol raw.xlsx')
SimpleLV.to_excel(writer, '510050')
writer.save()

fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y, Z = [], [], []  # strike, local vol, expiry
for i in range(len(col[1:])): # for each expiry
    X.extend(list(SimpleLV['Strike'])) # number of strikes
    Y.extend(len(list(SimpleLV['Strike'])) * [col[1:][i]])  # maturity is same for all strikes
    Z.extend(list(SimpleLV[col[1:][i]])) # local vol has a lot of strikes with nan
print(len(X))
print(len(Y))
print(len(Z))
plt.title('Raw Local Vol Points from Gatheral Mapping')
plt.xlabel('Strike ($)')
plt.ylabel('Maturity (days)')
#print(SimpleLV.count())
ax.scatter(X, Y, Z, s=9)
plt.show()

# Put
# Bullspread = put_matrix.diff(-1)
# Strikediff = -Bullspread['Strike']
# Bullspread = Bullspread.div(Strikediff, axis='index')
# Bullspread[Bullspread >= 0] = np.nan
# Bullspread[Bullspread < -3] = np.nan
# BullButterfly = Bullspread.diff(1).div(Strikediff, axis='index')
# Calender = put_matrix.diff(-1, axis=1)
# Calender = Calender.fillna(np.nan)
# Calender[Calender >= 0] = np.nan
# Calender = Calender.div(Timedelta, axis=1)
# BullButterfly[BullButterfly > 0] = np.nan
# denominator = 0.5 * -BullButterfly.multiply(K_sq_put, axis='index')
# denominator[denominator == 0] = np.nan
# BullspreadK = Bullspread.multiply(put_matrix['Strike'], axis='index')
# CminusKBS = put_matrix.sub(BullspreadK)
# RTminusDT = list(map(lambda x: inst_forward_rate(x) - div(x), TTM))
# Modifier = CminusKBS.multiply(RTminusDT)
# Numerator = (-Calender).sub(Modifier)
# Numerator[Numerator <= 0] = np.nan
# SimpleLVar = Numerator.div(denominator)
# SimpleLV = SimpleLVar.apply(lambda x: x ** 0.5)
# SimpleLV[SimpleLV < 0.05] = np.nan
# SimpleLV[SimpleLV > 1] = np.nan
# SimpleLV['Strike'] = put_matrix['Strike']

'''SimpleLV.to_excel(writer, 'NIKKA225P')
writer.save()'''


