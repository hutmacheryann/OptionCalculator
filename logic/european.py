"""
european.py

Implements pricing and Greeks for European call & put options 
under Black-Scholes framework
"""

from .option import Option
import numpy as np
from .black_scholes import BlackScholesModel
from scipy.stats import norm


class EuropeanOption(Option):
    """ 
    Represents a European-style option with Black-Scholes pricing.
    
    Attributes:
    S : float
        Current underlying price
    K : float
        Strike price
    T : float
        Time to maturity (years)
    r : float
        Risk-free rate
    sigma : 
        Volatility of the underlying
    q : float, optional
        Continuous dividend yield (default = 0)
    option_type :   str, optional
        'call' or 'put' (default = 'call')
    num_simulations :   int, optional
        Number of Monte Carlo simulations (default = 10000)
    num_steps : int, optional
        Time steps (default = 252)
    """

    def __init__(self, S, K, T, r, sigma, q=0, option_type='call', num_simulations=10000, num_steps=252):
        # Initialise parent Option attributes
        super().__init__(S, K, T, r, sigma, q, option_type, num_simulations, num_steps)
        # Create Black-Scholes model instance for analytical pricing
        self.bs_model = BlackScholesModel(self.S, self.K, self.T, self.r, self.sigma, self.q)
        # Compute d1 & d2 (used repeatedly for price & Greeks)
        self.d1 = self.bs_model.d1()
        self.d2 = self.bs_model.d2()
    
    def price(self):
       """Return analytical price using Black-Scholes model"""
       if self.option_type == 'call':
           return self.bs_model.call_price()
       else:
           return self.bs_model.put_price()

    def delta(self):
        """Sensitivity of option value to underlying price (∂V/∂S)"""

        if self.option_type == 'call':
            return norm.cdf(self.d1) * np.exp(-self.q * self.T)
        else:
            return (norm.cdf(self.d1) - 1) * np.exp(-self.q * self.T)

    def gamma(self):
        """Second derivative wrt price - curvature of option value"""

        if self.T <= 0:
            return 0
        
        # Same for call & put
        return (norm.pdf(self.d1) * np.exp(-self.q * self.T)) / (self.S * self.sigma * np.sqrt(self.T))

    def vega(self):
        """Sensitivity to volatility (∂V/∂σ), expressed per 1% change"""

        if self.T <= 0:
            return 0

        return self.S * norm.pdf(self.d1) * np.sqrt(self.T) * np.exp(-self.q * self.T) / 100

    def theta(self):
        """Time decay (∂V/∂t), per day"""

        if self.T <= 0:
            return 0

        if self.option_type == 'call':
            theta = (-(self.S * norm.pdf(self.d1) * self.sigma * np.exp(-self.q * self.T)) / (2 * np.sqrt(self.T))
                    - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2)
                    + self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(self.d1))
        else:
            theta = (-(self.S * norm.pdf(self.d1) * self.sigma * np.exp(-self.q * self.T)) / (2 * np.sqrt(self.T))
                    + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2)
                    - self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(-self.d1))

        return theta / 365 # convert annualised theta to per-day

    def rho(self):
        """Sensitivity to interest rate (∂V/∂r), expressed per 1% change"""

        if self.T <= 0:
            return 0

        if self.option_type == 'call':
            return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(self.d2) / 100
        else:
            return -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-self.d2) / 100

    def get_all_greeks(self):
        """Returns all major Greeks as dictionary"""

        return {
            'delta': self.delta(),
            'gamma': self.gamma(),
            'vega': self.vega(),
            'theta': self.theta(),
            'rho': self.rho()
        }