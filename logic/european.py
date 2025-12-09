"""
European Option - uses analytical Black-Scholes pricing and Greeks.
"""
from .option import Option
from .black_scholes import BlackScholesModel
from scipy.stats import norm
import numpy as np


class EuropeanOption(Option):
    """
    European option with closed-form Black-Scholes pricing.
    Overrides base class Greeks with analytical formulas.
    """

    def __init__(self, S, K, T, r, sigma, q=0, option_type='call', num_simulations=10000, num_steps=252):
        # Initialise parent Option attributes
        super().__init__(S, K, T, r, sigma, q, option_type, num_simulations, num_steps)
        # Create Black-Scholes model instance for analytical pricing
        self.bs_model = BlackScholesModel(S, K, T, r, sigma, q)
        # Compute d1 & d2 (used repeatedly for price & Greeks)
        self.d1 = self.bs_model.d1()
        self.d2 = self.bs_model.d2()

    def price(self):
        """Return analytical price using Black-Scholes model"""
        if self.option_type == 'call':
            return self.bs_model.call_price()
        return self.bs_model.put_price()

    # Analytical Greeks (override base class MC Greeks)

    def delta(self):
        """Calculate Delta: ∂V/∂S."""
        if self.option_type == 'call':
            return norm.cdf(self.d1) * np.exp(-self.q * self.T)
        return (norm.cdf(self.d1) - 1) * np.exp(-self.q * self.T)

    def gamma(self):
        """Calculate Gamma: ∂²V/∂S²."""
        if self.T <= 0:
            return 0
        return (norm.pdf(self.d1) * np.exp(-self.q * self.T)) / (self.S * self.sigma * np.sqrt(self.T))

    def vega(self):
        """Calculate Vega: ∂V/∂σ."""
        if self.T <= 0:
            return 0
        return self.S * norm.pdf(self.d1) * np.sqrt(self.T) * np.exp(-self.q * self.T)

    def theta(self):
        """Calculate Theta: ∂V/∂t."""
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
        return theta / 365

    def rho(self):
        """Calculate Rho: ∂V/∂r."""
        if self.T <= 0:
            return 0
        if self.option_type == 'call':
            return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(self.d2)
        return -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-self.d2)

    def get_all_greeks(self):
        """
        Calculate all Greeks and return as dictionary.

        Returns
        -------
        dict
            Greeks: {'delta', 'gamma', 'vega', 'theta', 'rho'}
        """
        return {
            'delta': self.delta(),
            'gamma': self.gamma(),
            'vega': self.vega(),
            'theta': self.theta(),
            'rho': self.rho()
        }