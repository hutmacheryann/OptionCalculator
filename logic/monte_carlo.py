import numpy as np
from .black_scholes import BlackScholesModel


class MonteCarloEngine:

    def __init__(self, num_simulations=10000, num_steps=252, seed=None):
        self.num_simulations = num_simulations
        self.num_steps = num_steps
        self.seed = seed
        # Use RandomState for reproducible, independent random number generation
        self.rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()

    def simulate_paths(self, S0, T, r, sigma, q=0):
        # Reset the random state to ensure consistent random numbers for Greek calculations
        if self.seed is not None:
            self.rng = np.random.RandomState(self.seed)

        dt = T / self.num_steps
        paths = np.zeros((self.num_simulations, self.num_steps + 1))
        paths[:, 0] = S0

        for t in range(1, self.num_steps + 1):
            Z = self.rng.standard_normal(size=self.num_simulations)
            paths[:, t] = paths[:, t-1] * np.exp((r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

        return paths

    def price_european(self, S0, K, T, r, sigma, q, option_type):
        paths = self.simulate_paths(S0, T, r, sigma, q)
        ST = paths[:, -1]

        if option_type.lower() == 'call':
            payoffs = np.maximum(ST - K, 0)
        else:
            payoffs = np.maximum(K - ST, 0)

        price = np.exp(-r * T) * np.mean(payoffs)
        return price

    def price_american(self, S0, K, T, r, sigma, q, option_type):
        """
        Price American option using Longstaff-Schwartz LSM algorithm.

        Reference: Longstaff & Schwartz (2001) "Valuing American Options by Simulation"
        """
        paths = self.simulate_paths(S0, T, r, sigma, q)
        dt = T / self.num_steps

        if option_type.lower() == 'call':
            payoff = np.maximum(paths - K, 0)
        else:
            payoff = np.maximum(K - paths, 0)

        # Value matrix: stores the value of continuing (not exercising) at each node
        value = payoff[:, -1].copy()

        # Backward induction
        for t in range(self.num_steps - 1, 0, -1):
            # Discount value from t+1 to t
            value = value * np.exp(-r * dt)

            # Regression only on ITM paths
            itm = payoff[:, t] > 0

            if np.sum(itm) > 0:
                # Regress discounted value on current stock price for ITM paths
                X = paths[itm, t]
                Y = value[itm]

                # Fit polynomial regression (degree 2)
                if len(X) >= 3:
                    regression = np.polyfit(X, Y, 2)
                    continuation = np.polyval(regression, X)
                else:
                    # Not enough points, use mean
                    continuation = np.full_like(Y, np.mean(Y))

                # Exercise if immediate payoff exceeds continuation value
                exercise_now = payoff[itm, t] > continuation

                # Update value: max of exercise now vs continue
                value[itm] = np.where(exercise_now, payoff[itm, t], value[itm])

        # Discount from t=1 to t=0
        value = value * np.exp(-r * dt)

        return np.mean(value)

    def price_asian(self, S0, K, T, r, sigma, q, option_type, average_type='arithmetic'):

        paths = self.simulate_paths(S0, T, r, sigma, q)

        if average_type == 'arithmetic':
            avg_prices = np.mean(paths, axis=1)
        else:
            avg_prices = np.exp(np.mean(np.log(paths), axis=1))

        if option_type.lower() == 'call':
            payoffs = np.maximum(avg_prices - K, 0)
        else:
            payoffs = np.maximum(K - avg_prices, 0)

        price = np.exp(-r * T) * np.mean(payoffs)
        return price

    def price_barrier(self, S0, K, T, r, sigma, q, option_type, barrier_type, barrier_level):

        paths = self.simulate_paths(S0, T, r, sigma, q)
        ST = paths[:, -1]

        if barrier_type == 'up-and-out':
            knocked = np.max(paths, axis=1) >= barrier_level
        elif barrier_type == 'up-and-in':
            knocked = np.max(paths, axis=1) >= barrier_level
        elif barrier_type == 'down-and-out':
            knocked = np.min(paths, axis=1) <= barrier_level
        elif barrier_type == 'down-and-in':
            knocked = np.min(paths, axis=1) <= barrier_level
        else:
            raise ValueError(f"Unknown barrier type: {barrier_type}")

        if option_type.lower() == 'call':
            payoffs = np.maximum(ST - K, 0)
        else:
            payoffs = np.maximum(K - ST, 0)

        if 'out' in barrier_type:
            payoffs = np.where(knocked, 0, payoffs)
        else:
            payoffs = np.where(knocked, payoffs, 0)

        price = np.exp(-r * T) * np.mean(payoffs)
        return price