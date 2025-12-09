"""
Monte Carlo Engine for option pricing.
"""
import numpy as np


class MonteCarloEngine:
    """
    Monte Carlo engine for simulating price paths and pricing options.

    Parameters
    ----------
    num_simulations : int, optional
        Number of simulation paths (default: 10000)
    num_steps : int, optional
        Time steps per path (default: 252)
    seed : int, optional
        Random seed for reproducibility (default: 42)
    """

    def __init__(self, num_simulations=10000, num_steps=252, seed=42):
        self.num_simulations = num_simulations
        self.num_steps = num_steps
        self.seed = seed
        # Create instance-specific random generator (not global state)
        self.rng = np.random.default_rng(seed)

    def simulate_paths(self, S0, T, r, sigma, q=0):
        """
        Generate price paths using Geometric Brownian Motion.

        Uses discretized GBM: S(t+dt) = S(t) * exp((r - q - σ²/2)dt + σ√dt·Z)

        Parameters
        ----------
        S0 : float
            Initial spot price
        T : float
            Time to maturity in years
        r : float
            Risk-free rate
        sigma : float
            Volatility
        q : float, optional
            Dividend yield (default: 0)

        Returns
        -------
        np.ndarray
            Price paths with shape (num_simulations, num_steps + 1)
        """
        dt = T / self.num_steps
        paths = np.zeros((self.num_simulations, self.num_steps + 1))
        paths[:, 0] = S0

        for t in range(1, self.num_steps + 1):
            Z = self.rng.standard_normal(self.num_simulations)
            paths[:, t] = paths[:, t-1] * np.exp(
                (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
            )
        return paths

    def reset_rng(self):
        """Reset the random generator to initial seed (for reproducibility)."""
        self.rng = np.random.default_rng(self.seed)

    def _lsm_pricing(self, paths, K, r, T, option_type):
        """
        Price American option using Longstaff-Schwartz Monte Carlo.

        Uses polynomial regression to estimate continuation values and
        determines optimal exercise decisions via backward induction.

        Parameters
        ----------
        paths : np.ndarray
            Simulated price paths
        K : float
            Strike price
        r : float
            Risk-free rate
        T : float
            Time to maturity
        option_type : str
            'call' or 'put'

        Returns
        -------
        float
            American option price
        """
        dt = T / self.num_steps

        if option_type.lower() == 'call':
            intrinsic_value = np.maximum(paths - K, 0)
        else:
            intrinsic_value = np.maximum(K - paths, 0)

        cash_flows = intrinsic_value[:, -1].copy()

        for t in range(self.num_steps - 1, 0, -1):
            discounted_cf = cash_flows * np.exp(-r * dt)
            itm = intrinsic_value[:, t] > 0

            if np.sum(itm) > 0:
                X = paths[itm, t]
                Y = discounted_cf[itm]
                regression = np.polyfit(X, Y, 2)
                continuation_value = np.polyval(regression, X)
                exercise = intrinsic_value[itm, t] > continuation_value
                cash_flows[itm] = np.where(exercise, intrinsic_value[itm, t], discounted_cf[itm])

        return np.exp(-r * dt) * np.mean(cash_flows)

    # ====================   Pricing Options   ====================

    def price_american(self, S0, K, T, r, sigma, q, option_type):
        """
        Price American option using LSM algorithm.

        Resets RNG for reproducibility before generating paths.
        """
        self.reset_rng()  # Ensure reproducibility
        paths = self.simulate_paths(S0, T, r, sigma, q)
        return self._lsm_pricing(paths, K, r, T, option_type)

    def price_asian(self, S0, K, T, r, sigma, q, option_type, average_type='arithmetic'):
        """
        Price Asian option with arithmetic or geometric averaging.

        Parameters
        ----------
        average_type : {'arithmetic', 'geometric'}
            Averaging method for path prices
        """
        self.reset_rng()  # Ensure reproducibility
        paths = self.simulate_paths(S0, T, r, sigma, q)

        if average_type == 'arithmetic':
            avg_prices = np.mean(paths, axis=1)
        else:
            avg_prices = np.exp(np.mean(np.log(paths), axis=1))

        if option_type.lower() == 'call':
            payoffs = np.maximum(avg_prices - K, 0)
        else:
            payoffs = np.maximum(K - avg_prices, 0)

        return np.exp(-r * T) * np.mean(payoffs)

    def price_barrier(self, S0, K, T, r, sigma, q, option_type, barrier_type, barrier_level):
        """
        Price Barrier option with knock-in/knock-out features.

        Parameters
        ----------
        barrier_type : str
            One of: 'up-and-out', 'up-and-in', 'down-and-out', 'down-and-in'
        barrier_level : float
            Barrier price level
        """
        self.reset_rng()  # Ensure reproducibility
        paths = self.simulate_paths(S0, T, r, sigma, q)
        ST = paths[:, -1]

        if barrier_type in ['up-and-out', 'up-and-in']:
            knocked = np.max(paths, axis=1) >= barrier_level
        else:
            knocked = np.min(paths, axis=1) <= barrier_level

        if option_type.lower() == 'call':
            payoffs = np.maximum(ST - K, 0)
        else:
            payoffs = np.maximum(K - ST, 0)

        if 'out' in barrier_type:
            payoffs = np.where(knocked, 0, payoffs)
        else:
            payoffs = np.where(knocked, payoffs, 0)

        return np.exp(-r * T) * np.mean(payoffs)