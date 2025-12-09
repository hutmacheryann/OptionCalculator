"""
Base option class for all option types.

Provides common functionality for pricing and Greeks calculation using
Monte Carlo simulation with Common Random Numbers (CRN).
"""
from .monte_carlo import MonteCarloEngine


class Option:

    """
    Abstract base class for all option types.

    Implements finite difference Greeks with CRN for variance reduction.
    Subclasses must implement: price(), _price_from_paths(), and theta().

    Parameters
    ----------
    S : float
        Current spot price (must be positive)
    K : float
        Strike price (must be positive)
    T : float
        Time to maturity in years
    r : float
        Risk-free rate (e.g., 0.05 for 5%)
    sigma : float
        Volatility (e.g., 0.2 for 20%)
    q : float, optional
        Dividend yield (default: 0)
    option_type : {'call', 'put'}
        Option type
    num_simulations : int, optional
        Number of MC paths (default: 10000)
    num_steps : int, optional
        Time steps per path (default: 252)
    seed : int, optional
        Random seed (default: 42)

    Notes
    -----
    Bump sizes are calibrated for LSM stability:
    - Delta: 0.5 (small relative to spot)
    - Gamma: 5.0 (larger for second derivative)
    - Vega: 0.01 (1% absolute vol change)
    - Rho: 0.01 (100 basis points)
    - Theta: 1/365 (one calendar day)
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
        """
        Calculate Theta: ∂V/∂T using time decay.

        Must be implemented by subclasses as optimal approach differs by type.

        Parameters
        ----------
        bump : float, optional
            Time bump in years (default: 1/365 = one calendar day)

        Returns
        -------
        float
            Theta (rate of time decay per year)
        """
        raise NotImplementedError("Subclasses must implement theta()")

    # === Path Generation ===

    def _get_paths_with_params(self, S=None, r=None, sigma=None):
        """Get paths with modified parameters. Resets RNG for CRN."""
        S = S if S is not None else self.S
        r = r if r is not None else self.r
        sigma = sigma if sigma is not None else self.sigma

        self.mc_engine.reset_rng()
        return self.mc_engine.simulate_paths(S, self.T, r, sigma, self.q)

    #============= Greeks =======================

    def delta(self, bump=0.5):
        """
        Calculate Delta: ∂V/∂S using central finite differences.

        Parameters
        ----------
        bump : float, optional
            Spot price bump (default: 0.5)

        Returns
        -------
        float
            Delta (typically 0 to 1 for calls, -1 to 0 for puts)
        """
        paths_up = self._get_paths_with_params(S=self.S + bump)
        paths_down = self._get_paths_with_params(S=self.S - bump)

        price_up = self._price_from_paths(paths_up)
        price_down = self._price_from_paths(paths_down)

        return (price_up - price_down) / (2 * bump)

    def gamma(self, bump=5.0):
        """
        Calculate Gamma: ∂²V/∂S² using central finite differences.

        Uses larger bump than Delta for stability with LSM pricing.

        Parameters
        ----------
        bump : float, optional
            Spot price bump (default: 5.0)

        Returns
        -------
        float
            Gamma (always positive for long options)
        """
        paths_up = self._get_paths_with_params(S=self.S + bump)
        paths_center = self._get_paths_with_params(S=self.S)
        paths_down = self._get_paths_with_params(S=self.S - bump)

        price_up = self._price_from_paths(paths_up)
        price_center = self._price_from_paths(paths_center)
        price_down = self._price_from_paths(paths_down)

        return (price_up - 2 * price_center + price_down) / (bump ** 2)

    def vega(self, bump=0.01):
        """
        Calculate Vega: ∂V/∂σ using central finite differences.

        Parameters
        ----------
        bump : float, optional
            Volatility bump (default: 0.01 = 1 percentage point)

        Returns
        -------
        float
            Vega (change in price per 1% vol increase)
        """
        paths_up = self._get_paths_with_params(sigma=self.sigma + bump)
        paths_down = self._get_paths_with_params(sigma=self.sigma - bump)

        price_up = self._price_from_paths(paths_up)
        price_down = self._price_from_paths(paths_down)

        return (price_up - price_down) / (2 * bump)

    def rho(self, bump=0.01):
        """
        Calculate Rho: ∂V/∂r using central finite differences.

        Parameters
        ----------
        bump : float, optional
            Rate bump (default: 0.01 = 100 basis points)

        Returns
        -------
        float
            Rho (change in price per 1% rate increase)
        """
        paths_up = self._get_paths_with_params(r=self.r + bump)
        paths_down = self._get_paths_with_params(r=self.r - bump)

        price_up = self._price_from_paths(paths_up)
        price_down = self._price_from_paths(paths_down)

        return (price_up - price_down) / (2 * bump)

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