"""
American Option - uses LSM pricing.
"""
from .option import Option


class AmericanOption(Option):
    """
    American option with early exercise capability.

    Pricing uses the Longstaff-Schwartz Least Squares Monte Carlo (LSM) algorithm.
    Greeks are calculated using finite differences with Common Random Numbers (CRN)
    for variance reduction.

    Inherits all attributes from Option base class.
    """

    def __init__(self, S, K, T, r, sigma, q=0, option_type='call', num_simulations=10000, num_steps=252, seed=42):
        # Initialise parent Option attributes
        super().__init__(S, K, T, r, sigma, q, option_type, num_simulations, num_steps, seed)

    def price(self):
        """
        Calculate American option price using LSM algorithm.

        Returns
        -------
        float
            Option price
        """
        self.mc_engine.reset_rng()
        return self.mc_engine.price_american(self.S, self.K, self.T, self.r, self.sigma, self.q, self.option_type)

    def _price_from_paths(self, paths):
        """Price from given paths (for Greeks)."""
        return self.mc_engine._lsm_pricing(paths, self.K, self.r, self.T, self.option_type)

    def theta(self, bump=1/365):
        """Theta: time decay (uses separate paths for T and T-bump)."""
        # Current price
        self.mc_engine.reset_rng()
        price_now = self.mc_engine.price_american(
            self.S, self.K, self.T, self.r, self.sigma, self.q, self.option_type
        )

        # Price at T - bump
        self.mc_engine.reset_rng()
        T_bump = max(self.T - bump, 1e-10)
        price_later = self.mc_engine.price_american(
            self.S, self.K, T_bump, self.r, self.sigma, self.q, self.option_type
        )

        return (price_later - price_now) / bump