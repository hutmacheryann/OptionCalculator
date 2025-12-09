"""
Base Option class - all MC options inherit from this.
Uses single seed and single num_simulations for consistency.
"""
from .monte_carlo import MonteCarloEngine


class Option:
    """
    Base class for all option types.

    This class provides common functionality and attributes shared by all option types.
    Subclasses should implement their specific pricing and Greeks calculation methods.

    Attributes
    ----------
    S : float
        Current underlying price
    K : float
        Strike price
    T : float
        Time to maturity (in years)
    r : float
        Risk-free interest rate
    sigma : float
        Volatility of the underlying asset
    q : float
        Continuous dividend yield (default: 0)
    option_type : str
        'call' or 'put'
    num_simulations : int
        Number of Monte Carlo simulations (default: 10000)
    num_steps : int
        Number of time steps in simulation (default: 252)
    mc_engine : MonteCarloEngine
        Monte Carlo simulation engine instance
    """

    def __init__(self, S, K, T, r, sigma, q=0, option_type='call',
                 num_simulations=10000, num_steps=252, seed=42):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
        self.option_type = option_type.lower()
        self.num_simulations = num_simulations
        self.num_steps = num_steps
        self.seed = seed

        self.mc_engine = MonteCarloEngine(num_simulations, num_steps, seed)

    # === Abstract Methods ===

    def price(self):
        """Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement price()")

    def _price_from_paths(self, paths):
        """Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _price_from_paths()")

    def theta(self, bump=1/365):
        """Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement theta()")

    # === Path Generation ===

    def _get_paths_with_params(self, S=None, r=None, sigma=None):
        """Get paths with modified parameters. Resets RNG for CRN."""
        S = S if S is not None else self.S
        r = r if r is not None else self.r
        sigma = sigma if sigma is not None else self.sigma

        self.mc_engine.reset_rng()
        return self.mc_engine.simulate_paths(S, self.T, r, sigma, self.q)

    # === Greeks (finite differences with CRN) ===
    # Note: Larger bumps reduce noise but increase bias
    # These defaults balance noise vs bias for LSM pricing

    def delta(self, bump=0.5):
        """Delta: ∂V/∂S"""
        paths_up = self._get_paths_with_params(S=self.S + bump)
        paths_down = self._get_paths_with_params(S=self.S - bump)

        price_up = self._price_from_paths(paths_up)
        price_down = self._price_from_paths(paths_down)

        return (price_up - price_down) / (2 * bump)

    def gamma(self, bump=5.0):
        """Gamma: ∂²V/∂S² (larger bump needed for LSM stability)"""
        paths_up = self._get_paths_with_params(S=self.S + bump)
        paths_center = self._get_paths_with_params(S=self.S)
        paths_down = self._get_paths_with_params(S=self.S - bump)

        price_up = self._price_from_paths(paths_up)
        price_center = self._price_from_paths(paths_center)
        price_down = self._price_from_paths(paths_down)

        return (price_up - 2 * price_center + price_down) / (bump ** 2)

    def vega(self, bump=0.01):
        """Vega: ∂V/∂σ"""
        paths_up = self._get_paths_with_params(sigma=self.sigma + bump)
        paths_down = self._get_paths_with_params(sigma=self.sigma - bump)

        price_up = self._price_from_paths(paths_up)
        price_down = self._price_from_paths(paths_down)

        return (price_up - price_down) / (2 * bump)

    def rho(self, bump=0.01):
        """Rho: ∂V/∂r"""
        paths_up = self._get_paths_with_params(r=self.r + bump)
        paths_down = self._get_paths_with_params(r=self.r - bump)

        price_up = self._price_from_paths(paths_up)
        price_down = self._price_from_paths(paths_down)

        return (price_up - price_down) / (2 * bump)

    def get_all_greeks(self):
        """Calculate all Greeks."""
        return {
            'delta': self.delta(),
            'gamma': self.gamma(),
            'vega': self.vega(),
            'theta': self.theta(),
            'rho': self.rho()
        }