import matplotlib.pyplot as plt
import numpy as np
import IPython
import BSM_class
import random
import pandas as pd
IPython.get_ipython().magic("matplotlib auto")  # show interactive plot not inline



class European_Call_Payoff:

    def __init__(self, strike):
        self.strike = strike

    def get_payoff(self, stock_price):
        if stock_price > self.strike:
            return stock_price - self.strike
        else:
            return 0


class UpandOutBarrier_European_Call_Payoff:

    def __init__(self, strike, barrier_level1=0, barrier_level2=0, barrier_level3=0, barrier_level4=0, barrier_level5=0, barrier_level6=0, rebate=0):
        # default values are set so all inputs are optional
        self.strike = strike
        self.rebate = rebate
        self.barrier_level1 = barrier_level1
        self.barrier_level2 = barrier_level2
        self.barrier_level3 = barrier_level3
        self.barrier_level4 = barrier_level4
        self.barrier_level5 = barrier_level5
        self.barrier_level6 = barrier_level6
        self.barrier_trigger = False

    def check_barrier1(self, stock_price):
        if stock_price > self.barrier_level1:
            self.barrier_trigger = True

    def check_barrier2(self, stock_price):
        if stock_price > self.barrier_level2:
            self.barrier_trigger = True

    def check_barrier3(self, stock_price):
        if stock_price > self.barrier_level3:
            self.barrier_trigger = True

    def check_barrier4(self, stock_price):
        if stock_price > self.barrier_level4:
            self.barrier_trigger = True

    def check_barrier5(self, stock_price):
        if stock_price > self.barrier_level5:
            self.barrier_trigger = True

    def check_barrier6(self, stock_price):
        if stock_price > self.barrier_level6:
            self.barrier_trigger = True

    def get_payoff(self, stock_price):
        if not self.barrier_trigger:
            if stock_price > self.strike:
                return stock_price - self.strike
            else:
                return 0
        else:
            return self.rebate


class GeometricBrownianMotion:

    # constructor
    def __init__(self, initial_price, drift, volatility, dt, T):
        self.current_price = initial_price
        self.initial_price = initial_price
        self.drift = drift
        self.volatility = volatility
        self.dt = dt
        self.T = T
        self.prices = []
        self.simulate_paths()

    # will run whenever class object is initiated
    def simulate_paths(self):
        while (self.T - self.dt > 0):
            # dWt = np.random.normal(0, np.sqrt(self.dt))  # That is the Brownian motion by definition
            Z = np.random.normal(0, 1)
            logreturn = (self.drift - 0.5 * self.volatility ** 2) * self.dt + self.volatility * Z * np.sqrt(dt)
            # Note: sigma*Z*sqrt(dt)=sigma*dW(t) they are the same thing by definitions
            self.current_price = np.exp(np.log(self.current_price) + logreturn)  # ln(new)=ln(stock price last)+logreturn
            self.prices.append(self.current_price)  # Append new price to series
            self.T -= self.dt  # Account for the step in time


def main(rebate):

    # Generate a set of sample brownian motion paths
    for i in range(0, paths):
        price_paths.append(GeometricBrownianMotion(initial_price, drift, volatility, dt, T).prices)

    # Plot all brownian motion paths, optional
    plt.figure(figsize=(16, 8))
    for price_path in price_paths:
        plt.plot(price_path)
    plt.xlabel('steps/days')
    plt.ylabel('stock price')
    plt.grid()
    #plt.show()

    ec = European_Call_Payoff(strike_price)
    vanilla_call_payoffs = []
    case1_call_payoffs = []
    case2_call_payoffs = []
    case3_call_payoffs = []
    case41_call_payoffs = []
    case42_call_payoffs = []
    case43_call_payoffs = []

    # generate payoff in each brownian motion
    for price_path in price_paths:

        # initiate here due to internal class flags need to be reset for each price_path
        bc_case1 = UpandOutBarrier_European_Call_Payoff(strike_price, 110, rebate=rebate)  # 110 is the barrier price
        bc_case2 = UpandOutBarrier_European_Call_Payoff(strike_price, 110, 108, rebate=rebate)
        bc_case3 = UpandOutBarrier_European_Call_Payoff(strike_price, 105, 106, 107, 108, 109, 110, rebate=rebate)
        bc_case41 = UpandOutBarrier_European_Call_Payoff(strike_price, 110, rebate=rebate)
        bc_case42 = UpandOutBarrier_European_Call_Payoff(strike_price, 110, rebate=rebate)
        bc_case43 = UpandOutBarrier_European_Call_Payoff(strike_price, 110, rebate=rebate)

        # last three month check barrier
        for price in price_path[-90:]: # this is in chronological order
            bc_case1.check_barrier1(price)
            bc_case2.check_barrier1(price)
            # No need 'break' because once a class flag is set, it cannot be reversed unless re-initiate class

        # first 3 month check barrier
        for price in price_path[:90]: # this is in chronological order
            bc_case2.check_barrier2(price)

        # for linear barrier setup
        for price in price_path[:30]:
            bc_case3.check_barrier1(price)
        for price in price_path[30:60]:
            bc_case3.check_barrier2(price)
        for price in price_path[60:90]:
            bc_case3.check_barrier3(price)
        for price in price_path[90:120]:
            bc_case3.check_barrier4(price)
        for price in price_path[120:150]:
            bc_case3.check_barrier5(price)
        for price in price_path[150:180]:
            bc_case3.check_barrier6(price)

        # monthly check barrier
        for price in price_path[::30]:
            bc_case41.check_barrier1(price)

        # weekly check barrier
        for price in price_path[::7]:
            bc_case42.check_barrier1(price)

        # daily check barrier
        for price in price_path[:]:
            bc_case43.check_barrier1(price)

        # under each path, discount the final payoff to present time
        case1_call_payoffs.append(bc_case1.get_payoff(price_path[-1]) / np.exp(risk_free_rate * T))
        case2_call_payoffs.append(bc_case2.get_payoff(price_path[-1]) / np.exp(risk_free_rate * T))
        case3_call_payoffs.append(bc_case3.get_payoff(price_path[-1]) / np.exp(risk_free_rate * T))
        case41_call_payoffs.append(bc_case41.get_payoff(price_path[-1]) / np.exp(risk_free_rate * T))
        case42_call_payoffs.append(bc_case42.get_payoff(price_path[-1]) / np.exp(risk_free_rate * T))
        case43_call_payoffs.append(bc_case43.get_payoff(price_path[-1]) / np.exp(risk_free_rate * T))
        vanilla_call_payoffs.append(ec.get_payoff(price_path[-1]) / np.exp(risk_free_rate*T))

    trade = BSM_class.OptionTrade(initial_price, strike_price, risk_free_rate, volatility, T, dividend_yield, 1)
    # print('European Option BSM price is: ', trade.BSM())
    # print('European Vanilla Option simulation price is: ', np.average(vanilla_call_payoffs))
    # print('European Case1 Option simulation price is: ', np.average(case1_call_payoffs))
    # print('European Case2 Option simulation price is: ', np.average(case2_call_payoffs))
    # print('European Case3 Option simulation price is: ', np.average(case3_call_payoffs))
    # print('European Case4_monthly Option simulation price is: ', np.average(case41_call_payoffs))
    # print('European Case4_weekly Option simulation price is: ', np.average(case42_call_payoffs))
    # print('European Case4_daily Option simulation price is: ', np.average(case43_call_payoffs))
    # BSM will match brownian motion monte carlo pricing if drift is zero!
    # no one is better than the other, they are just different approaches

    return [trade.BSM(),np.average(vanilla_call_payoffs),np.average(case1_call_payoffs),np.average(case2_call_payoffs),np.average(case3_call_payoffs),np.average(case41_call_payoffs),np.average(case42_call_payoffs),np.average(case43_call_payoffs)]


if __name__ == "__main__":

    # Model Parameters

    random.seed(1)
    paths = 1000
    initial_price = 100.0
    strike_price = 105.0
    volatility = 0.3
    dt = 1 / 365 # steps
    T = 0.5  # 6 month maturity
    risk_free_rate = 0.02
    drift = risk_free_rate
    dividend_yield = 0.0
    price_paths = [] # contains set of Brownian motions

    # Main

    df = pd.DataFrame([],index=['BSM price','vanilla price','constant barrier price','step-up barrier price','linear barrier price','discrete barrier monthly','discrete barrier weekly','discrete barrier daily'])
    for i in [0,4,8]:
        df['rebate=$'+str(i)]=main(i)

display(df)