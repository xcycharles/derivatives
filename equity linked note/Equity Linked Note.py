import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
from scipy import stats, optimize, interpolate
import math
import time


class Correlated_Path:

    '''
    Use 26 weekly correlation
    Z = np.random.normal(0,1,size=(3,1))
    this class will be inherited by GBM, so need to input random part there
    '''

    def get(Z):
        corr_matrix = pd.read_csv('26w corr matrix.csv', header=None).iloc[0:3, 0:3]  # matrix的顺序是700 5 941
        L = np.linalg.cholesky(corr_matrix)

        return np.matmul(L, Z).reshape(3)

        # note dimension should be (3,) not (1,3) by using transpose
        # need to reshape to (3,) because it will be used for scalar/dot multiplication in GBM class with [1,2,3]


class Geometric_Brownian_Motion:
    '''
    get single stochastic path
    initial_price = [700,5,941], 5/17 EOD
    drift = fixed risk free rate
    volatility = [700,5,941]
    dividend yield = [700,5,941]
    dt = 1/365
    T = maturity
    dividend yield = 0
    Z = Correlated_Path.get(np.random.normal(0,1,size=(3,1)))
    vol_flag=0: use constant vol
    vol_flag=1: use function local_vol_***(St,S0,T)
    '''

    def __init__(self, initial_price, drift, volatility, dt, T, dividend_yield, vol_flag):

        self.current_price_list = np.array(initial_price)  # use array to do scalar multiplication
        self.initial_price_list = np.array(initial_price)
        self.dividend_yield = np.array(dividend_yield)
        self.drift = drift  # fixed interest rate for now
        self.dt = dt
        self.maturity = T
        self.time = 0
        self.prices = []  # to save all previous price
        self.vols = []  # to save all the local vols
        self.vol_flag = vol_flag
        self.simulate_paths()  # automatically initialized
        self.get_volatility()  # automatically initialized

    def get_volatility(self):

        if self.vol_flag == 0:
            self.volatility_list = np.array(volatility)
        elif self.vol_flag == 1:
            self.volatility_list = [local_vol_700(self.current_price_list[0], self.initial_price_list[0], self.time), \
                                    local_vol_5(self.current_price_list[1], self.initial_price_list[1], self.time), \
                                    local_vol_941(self.current_price_list[2], self.initial_price_list[2], self.time)]

    def simulate_paths(self):

        while (self.time < self.maturity):
            self.get_volatility()
            dSt = (self.drift + self.dividend_yield) * self.dt * self.current_price_list \
                  + self.volatility_list * Correlated_Path.get(np.random.normal(0, 1, size=(3, 1))) \
                  * np.sqrt(dt) * self.current_price_list
            self.current_price_list = self.current_price_list + dSt  # 1 by 3 dimension
            self.prices.append(self.current_price_list)  # Append new price to series for every dt
            self.vols.append(self.volatility_list)
            self.time += self.dt  # Account for the step in time

    # eventually get prices as 1 full stochastic path from 0 to T: (365 * 3 stocks)


class Payoff:

    """
    股票挂钩票据/高息票据/股权连结商品
    If Snl >= Sol, payoff at maturitGet payoff according to term sheet
    If Snl < Sol, payoff at maturityWe need to solve for PR such that the note price is 98% of the issue price
    only care about last maturity
    Investor wins of the worst performing stock in the basket outperforms itself at maturity

    disount = 0.98
    denomination = 10000
    discount_factor = e^(-rT)
    Sn = closing price of share on valuation date
    S0 = initial price of share
    snso1/2/3 = Sn/S0 = laggard share definition which is the lowest value fo Sn/S0
    Snl = closing price of laggard share on valuation date
    Sol = initial price of laggard share
    """

    def Equity_Linked_Note(pr_guess):
        global df  # need this throughout for analysis
        df = pd.DataFrame(data=price_paths)
        df['snso1'] = df.iloc[:, 0] / initial_price[0]
        df['snso2'] = df.iloc[:, 1] / initial_price[1]
        df['snso3'] = df.iloc[:, 2] / initial_price[2]
        df['min'] = df.iloc[:, 3:6].min(axis=1) # min ratio
        df['idxmin'] = df.iloc[:, 3:6].idxmin(axis=1) # find min ratio
        df['sol'] = df['idxmin'].replace({'snso1': initial_price[0], 'snso2': initial_price[1], 'snso3': initial_price[2]})
        df['snl'] = df['min'] * df['sol']
        df['snl/sol'] = df['snl'] / df['sol']
        # below is when pr_guess=1
        df['payoff'] = np.where(df['snl'] >= df['sol'], denomination*(1+pr_guess*(df['snl/sol']-1)),
                                denomination*df['snl/sol'].apply(lambda x: x if x > 0.9 else 0.9))
        
    def pr_optimize_method1(pr_guess, discount=0.98):
        # for optimization must put in a variable not dataframe column
        payoff_opt = np.where(df['snl'] >= df['sol'], denomination*(1+pr_guess*(df['snl/sol']-1)),
                              denomination*df['snl/sol'].apply(lambda x: x if x > 0.9 else 0.9))
        return payoff_opt.mean() * discount_factor - discount * denomination

    # should not do row by row optimization because payoff is not a direct function of pr
    # maybe can use heavyside distribution?
    # def pr_optimize_method2(pr_guess, discount=0.98): # problem is doesnt converge in optimization
    #     for index in range(0,len(df)):
    #         payoff = np.where(df['snl'][index]>=df['sol'][index],pr_guess*(df['snl/sol'][index]-1),max(df['snl/sol'][index],0.9))
    #         fun = lambda pr_guess: payoff*denomination*discount_factor-discount*denomination
    #         df['pr_hybrid'][index] = optimize.root(fun, pr_guess, method='hybr').x
    #         df['pr_bisect'][index] = optimize.bisect(fun, -1000, 1000)
    #         df['pr_newton'][index] = optimize.newton(fun, pr_guess)


class Monte_Carlo:
    '''
    price_path: just containing the maturity price
    price_path_full = containing full path for all simulations for plotting purpose
    '''

    def __init__(self, pathnumber):
        self.pathnumber = pathnumber
        self.generate()  # automatically run this

    def generate(self):
        global price_paths  # for save time not have to initiate again
        global price_paths_full  # for plotting monte carlo
        global vol_paths  # for plotting
        price_paths = []
        price_paths_full = []
        vol_paths = []
        for i in range(0, self.pathnumber):
            # must instantiate GBM class here for each monte carlo simulation
            GBM = Geometric_Brownian_Motion(initial_price, drift, volatility, dt, T, dividend_yield, vol_flag)
            price_paths_full.append(GBM.prices)
            vol_paths.append(GBM.vols)
        vol_paths = np.array(vol_paths)  # array is easier for later things
        price_paths_full = np.array(price_paths_full)  # paths * all dt * number stocks
        price_paths = price_paths_full[:, -1, :]  # all possible last price, change from (100, 365, 3) to (100, 3) or (simulations * stock)

        # no need to return anything because price_paths was defined as a global variable
        pass

########################################################
# Model Parameters

# list order for stocks is [700,5,941]
random.seed(1)
volatility = [0.4372, 0.32091, 0.3509]  # if constant vol use 5/17 26w/130d historical vol
pathnumber = 1000  # around 3 min for 100 runs
initial_price = [600.5, 48.55, 48.8] # on 5/17/2021
denomination = 10000.0
issue_price = 1.00 * 10000.0
dt = 1 / 250  # each step
T = 1  # 1 year maturity
risk_free_rate = 0.0008  # 1 month annualized HIBOR https://www.hangseng.com/en-hk/personal/mortgages/rates/hibor/
drift = risk_free_rate
discount_factor = np.exp(-risk_free_rate * T)
dividend_yield = [0.002660903, 0.0, 0.068118172]  # one year future dividend yield gotten from div swap in bbg
pr_guess = 1  # initial guess for participation rate
# it is the percentage amount investor participates in the appreciation of the underlying equity
# If the participation rate is 100%, then a 5% increase in the underlying is a 5% increase for the eventual payout on the note.
# Generally, the participation rate is better in longer maturity notes, since the total amount of interest given up by the investor is higher.
vol_flag = 0  # 0 is constant vol, 1 is local vol

##############################################################
# main

def main():

    # Geometric_Brownian_Motion
    # GBM class will be instantiated as 1 full stochastic path from 0 to T: (365 * 3 stocks) in Monte_Carlo class

    Monte_Carlo(pathnumber)
    # populate entire price_paths array as number of simulations * last price at maturity * 3 stocks

    Payoff.Equity_Linked_Note(pr_guess)
    # generate corresponding dataframe before payoff and pr optimization

    result.append(optimize.newton(Payoff.pr_optimize_method1, pr_guess))
    #result.append(pr_hybrid = optimize.root(Payoff.pr_optimize_method1, pr_guess, method='hybr').x)  # modified Powell method
    #result.append(optimize.bisect(Payoff.pr_optimize_method1, -1000, 1000))

############################################################
# run

start_time = time.time()
# each monte carlo only does 1 optimization, difference being 3x+1=0 or 9999x+142783=0
# therefore need to average the optimized value in another loop
result = []
for i in range(15):
    try:
        main()
    except:
        pass
print(f'Participation Rate is {np.mean(result)} for pricing at 98% of issue price')
print(f"%mean of payoff when pr is 1: {df['payoff'].mean()}")  # should be 1
print("time elapsed: {:.2f}m".format((time.time() - start_time) / 60))

#############################################################
# plot Monte Carlo

for j in range(0, 1):  # stock 1-3
    plt.figure(figsize=(10, 5))
    for i in range(0, 20):  # price simulations
        plt.title('700 price')
        plt.grid()
        plt.xlabel('dt')
        plt.ylabel('spot price')
        plt.ylim(0, 1800)
        plt.plot(price_paths_full[i, :, j])
    plt.figure(figsize=(10, 5))
    for i in range(0, 20):  # vol simulations
        plt.title('700 vol')
        plt.grid()
        plt.xlabel('dt')
        plt.ylabel('vol')
        plt.axhline(y=0.4372, color='red', alpha=0.1)
        plt.ylim(0, 0.6)
        plt.plot(vol_paths[i, :, j])  # simulation * dt * stock
plt.show()
