"""
Barrier Option - knock-in/knock-out features.

Option activates or deactivates when price crosses barrier level.
"""
from .option import Option
import numpy as np


class BarrierOption(Option):
    """
    Barrier option with knock-in or knock-out features.

    Greeks calculated via finite differences with CRN.

    Parameters
    ----------
    S, K, T, r, sigma, q : float
        Standard option parameters (see Option base class)
    option_type : {'call', 'put'}
        Option type
    barrier_type : str
        One of: 'down-and-out', 'down-and-in', 'up-and-out', 'up-and-in'
    barrier_level : float
        Price level that activates/deactivates the option
    num_simulations : int, optional
        Number of MC paths (default: 10000)
    num_steps : int, optional
        Time steps per path (default: 252)
    seed : int, optional
        Random seed (default: 42)
    """

    def __init__(self, S, K, T, r, sigma, q=0, option_type='call', barrier_type='down-and-out', barrier_level=None, num_simulations=10000, num_steps=252, seed=42):
        # Initialise parent Option attributes
        super().__init__(S, K, T, r, sigma, q, option_type, num_simulations, num_steps, seed)
        # Two barrier specific parameter
        self.barrier_type = barrier_type
        self.barrier_level = barrier_level

        if barrier_level is None:
            raise ValueError("barrier_level is required")

    def price(self):
        """
        Calculate Barrier option price using Monte Carlo.

        Returns
        -------
        float
            Option price
        """
        self.mc_engine.reset_rng()
        return self.mc_engine.price_barrier(self.S, self.K, self.T, self.r, self.sigma, self.q, self.option_type, self.barrier_type, self.barrier_level)

    def _price_from_paths(self, paths):
        """Price from given paths (for Greeks)."""
        ST = paths[:, -1]

        if self.barrier_type in ['up-and-out', 'up-and-in']:
            knocked = np.max(paths, axis=1) >= self.barrier_level
        else:
            knocked = np.min(paths, axis=1) <= self.barrier_level

        if self.option_type == 'call':
            payoffs = np.maximum(ST - self.K, 0)
        else:
            payoffs = np.maximum(self.K - ST, 0)

        if 'out' in self.barrier_type:
            payoffs = np.where(knocked, 0, payoffs)
        else:
            payoffs = np.where(knocked, payoffs, 0)

        return np.exp(-self.r * self.T) * np.mean(payoffs)

    def theta(self, bump=1/365):
        """Theta: time decay (uses separate paths for T and T-bump)."""
        self.mc_engine.reset_rng()
        price_now = self.mc_engine.price_barrier(
            self.S, self.K, self.T, self.r, self.sigma, self.q,
            self.option_type, self.barrier_type, self.barrier_level
        )

        self.mc_engine.reset_rng()
        T_bump = max(self.T - bump, 1e-10)
        price_later = self.mc_engine.price_barrier(
            self.S, self.K, T_bump, self.r, self.sigma, self.q,
            self.option_type, self.barrier_type, self.barrier_level
        )

        return (price_later - price_now) / bump