"""
Asian Option - path-dependent averaging.
"""
from .option import Option
import numpy as np


class AsianOption(Option):

    """
    Asian option with payoff based on average underlying price.

    Supports both arithmetic and geometric averaging. Greeks are calculated
    using finite differences with Common Random Numbers (CRN).

    Attributes
    ----------
    average_type : str
        'arithmetic' or 'geometric' averaging method
    """
    def __init__(self, S, K, T, r, sigma, q=0, option_type='call', average_type='arithmetic', num_simulations=10000, num_steps=252, seed=42):
        # Initialise parent Option attributes
        super().__init__(S, K, T, r, sigma, q, option_type, num_simulations, num_steps, seed)
        # Asian specific paramter
        self.average_type = average_type

    def price(self):
        """
        Calculate Asian option price using Monte Carlo.

        Returns
        -------
        float
            Option price
        """
        self.mc_engine.reset_rng()
        return self.mc_engine.price_asian(self.S, self.K, self.T, self.r, self.sigma, self.q, self.option_type, self.average_type)

    def _price_from_paths(self, paths):
        """Price from given paths (for Greeks)."""
        if self.average_type == 'arithmetic':
            avg_prices = np.mean(paths, axis=1)
        else:
            avg_prices = np.exp(np.mean(np.log(paths), axis=1))

        if self.option_type == 'call':
            payoffs = np.maximum(avg_prices - self.K, 0)
        else:
            payoffs = np.maximum(self.K - avg_prices, 0)

        return np.exp(-self.r * self.T) * np.mean(payoffs)

    def theta(self, bump=1/365):
        """Theta: time decay (uses separate paths for T and T-bump)."""
        self.mc_engine.reset_rng()
        price_now = self.mc_engine.price_asian(
            self.S, self.K, self.T, self.r, self.sigma, self.q,
            self.option_type, self.average_type
        )

        self.mc_engine.reset_rng()
        T_bump = max(self.T - bump, 1e-10)
        price_later = self.mc_engine.price_asian(
            self.S, self.K, T_bump, self.r, self.sigma, self.q,
            self.option_type, self.average_type
        )

        return (price_later - price_now) / bump