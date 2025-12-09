"""
Monte Carlo Engine for option pricing.
"""
import numpy as np


class MonteCarloEngine:
    """Monte Carlo simulation engine for option pricing."""

    def __init__(self, num_simulations=10000, num_steps=252, seed=42):
        self.num_simulations = num_simulations
        self.num_steps = num_steps
        self.seed = seed
        # Create instance-specific random generator (not global state)
        self.rng = np.random.default_rng(seed)

    def simulate_paths(self, S0, T, r, sigma, q=0):
        """Generate price paths using GBM dynamics."""
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
        """Price American option using LSM algorithm."""
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
        """Price American option using LSM."""
        self.reset_rng()  # Ensure reproducibility
        paths = self.simulate_paths(S0, T, r, sigma, q)
        return self._lsm_pricing(paths, K, r, T, option_type)

    def price_asian(self, S0, K, T, r, sigma, q, option_type, average_type='arithmetic'):
        """Price Asian option."""
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
        """Price Barrier option."""
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