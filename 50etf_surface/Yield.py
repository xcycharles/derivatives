import pandas as pd
import datetime
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.misc import derivative

'''This is a script on calculating the interest rate 
and dividend yield curve, it takes the yield curve and dividend yield futures
as input and then we defined a function spotrate() and dividend(),
 which input an index name and output a function of the country's interest rate
 and dividend. These functions then input a time and output an interest rate 
 or dividend yield at that point of time continuously'''

# http://yield.chinabond.com.cn/#
xls_yield = pd.ExcelFile('Yield Curve CN.xlsx')
CN = xls_yield.parse('CN')
# 股息收益率 = 最近12个月的现金分红总额 / A股总市值。
# https://wglh.com/chinaindicesdividend/sh000016/
# no equity etf dividend future/swap products in China, so assume constant
# note after equity option pay div, contract multiplier and strikes will change, resulting somewhat less liquid option chain on same expiry
div_yield_china = 0.024

China = CN[['YTM', 'TTM']].copy()
China['YTM'] = China['YTM'].div(100, axis='index')
# convert from EAR to continuous compound rate
China['YTM'] = China['YTM'].apply(np.log1p)  # （1+EAR)=e^r

x0 = np.array(China['TTM'])
y0 = np.array(China['YTM'])

cn = interpolate.CubicSpline(x0, y0)

plt.plot(x0, y0, 'bo', label='data point')
plt.plot(x0, cn(x0), 'r-', label='cubic spilne')
plt.title('China zero cupon gov bond yield curve')
plt.xlabel('year')
plt.ylabel('YTM')
plt.legend()
plt.show()

def spotrate(x):
    if x == "CN":
        return cn
