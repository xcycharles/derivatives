import numpy as np
from scipy import stats

class OptionTrade:
    def __init__(self, stock_price=100, strike_price=100, risk_free_rate=0.02, volatility=0.2, time_to_maturity=0.25, dividend_yield=0.03, iscall=1):
        '''
        S = current spot price
        E = strike price
        T = time to maturity expressed as fraction of days per year
        r = annual risk free rate
        q = annual dividend yield
        sigma = annual volatility
        '''
        self.S = stock_price
        self.E = strike_price
        self.r = risk_free_rate
        self.sigma = volatility
        self.T = time_to_maturity
        self.q = dividend_yield
        self.iscall = iscall
        '''
        Calculate option pricing using Black Scholes
        https://www.macroption.com/black-scholes-formula/
        '''
    def BSM(self):
        self.d1 = (np.log(self.S / self.E) + self.T * (self.r - self.q + np.power(self.sigma, 2) / 2)) / (self.sigma * np.sqrt(self.T))
        self.d2 = self.d1 - self.sigma * np.sqrt(self.T)
        if self.iscall:
            self.price = self.S * np.exp(-self.q * self.T) * stats.norm.cdf(self.d1) - self.E * np.exp(-self.r * self.T) * stats.norm.cdf(self.d2)
        else:
            self.price = self.E * np.exp(-self.r * self.T) * stats.norm.cdf(-self.d2) - self.S * np.exp(-self.q * self.T) * stats.norm.cdf(-self.d1)

        return self.price
