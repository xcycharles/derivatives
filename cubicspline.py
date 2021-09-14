import pandas as pd
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from sympy import Symbol, diff, erf, sqrt, init_printing, ln, exp, diff, lambdify

spot = 3.24
strike = [3.1,3.2,3.3,3.4,3.5,3.6,3.7]
vol = [0.3,0.2,0.1,0.2,0.26,0.39,0.56]
# plt.plot(strike,vol,label='original')
# plt.legend()


splineobj = interpolate.CubicSpline(strike, vol)
splinevol = splineobj(strike)
plt.plot(strike,splinevol,label='cubicspline implied vol vs strike')
plt.legend()


# num = np.arange(3.1,3.7,0.1)
# splineobj = interpolate.CubicSpline(strike, vol)
# splinevol = splineobj(num)
# plt.plot(num,splinevol,label='cubicspline interpolation')
# plt.legend()


splineobj = interpolate.CubicSpline(strike, vol)
splinevol1 = splineobj(strike,1)
# plt.plot(strike,splinevol,label='cubicspline 1st d')
# plt.legend()
# plt.show()
splinevol2 = splineobj(strike,2)
# plt.plot(strike,splinevol,label='cubicspline 2nd d')
# plt.legend()
# plt.show()
splinevol3 = splineobj(strike,3)

taylor = list(map(lambda x:splineobj(spot,0)+splineobj(spot,1)*(x-spot)
                +0.5*splineobj(spot,nu=2)*np.power((x-spot),2)
                #+(1/6)*splineobj(spot,3)*np.power((x-spot),3)
                ,strike))
plt.plot(strike,taylor,label='taylor')
plt.legend()
plt.show()