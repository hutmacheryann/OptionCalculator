import numpy as np
import scipy
from scipy.stats import norm


class BlackScholesModel:

    def __init__(self, S, K, T, r, sigma, q=0):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
    
    def d1(self):
        if self.T <= 0: return 0
        return (np.log(self.S / self.K) + (self.r - self.q + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))

    def d2(self):
        if self.T <= 0: return 0
        return self.d1(self.S, self.K, self.T, self.r, self.sigma, self.q) - self.sigma * np.sqrt(self.T)

    def call_price(self):
        if self.T <= 0: return max(self.S - self.K, 0)
        return self.S * np.exp(-self.q * self.T) * norm.cdf(self.d1()) - self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2())

    def put_price(self):
        if self.T <= 0: return max(self.K - self.S, 0)
        return self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2()) - self.S * np.exp(-self.q * self.T) * norm.cdf(-self.d1())

    @staticmethod
    def simulate_paths(S0, T, r, sigma, q, num_simulations, num_steps):

        dt = T / num_steps
        paths = np.zeros((num_simulations, num_steps + 1))
        paths[:, 0] = S0

        for t in range(1, num_steps + 1):
            Z = np.random.standard_normal(size=num_simulations)
            paths[:, t] = paths[:, t-1] * np.exp((r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

        return paths