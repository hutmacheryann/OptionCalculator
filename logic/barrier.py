from .option import Option
import numpy as np
from scipy.stats import norm


class BarrierOption(Option):

    def __init__(self, S, K, T, r, sigma, q=0, option_type='call', num_simulations=10000, num_steps=252, barrier_type='down-and-out', barrier_level=None):
        super().__init__(S, K, T, r, sigma, q, option_type, num_simulations, num_steps)
        self.barrier_type = barrier_type
        self.barrier_level = barrier_level

    def price(self):
        return self.mc_engine.price_barrier(
            self.S, self.K, self.T, self.r, self.sigma, self.q,
            self.option_type, self.barrier_type, self.barrier_level
        )
    
    def price_closed_form(self):

        if self.barrier_type == 'down-and-out' and self.option_type == 'call':
            H = self.barrier_level
            if self.S <= H:
                return 0

            lambda_val = (self.r - self.q + 0.5 * self.sigma**2) / (self.sigma**2)
            y = np.log(H**2 / (self.S * self.K)) / (self.sigma * np.sqrt(self.T)) + lambda_val * self.sigma * np.sqrt(self.T)
            x1 = np.log(self.S / H) / (self.sigma * np.sqrt(self.T)) + lambda_val * self.sigma * np.sqrt(self.T)
            y1 = np.log(H / self.S) / (self.sigma * np.sqrt(self.T)) + lambda_val * self.sigma * np.sqrt(self.T)

            if self.K > H:
                from .black_scholes import BlackScholesModel
                bs = BlackScholesModel()
                vanilla_call = bs.call_price(self.S, self.K, self.T, self.r, self.sigma, self.q)

                correction = (self.S * np.exp(-self.q * self.T) * (H / self.S)**(2 * lambda_val) * norm.cdf(y) -
                             self.K * np.exp(-self.r * self.T) * (H / self.S)**(2 * lambda_val - 2) * norm.cdf(y - self.sigma * np.sqrt(self.T)))

                return vanilla_call - correction
            else:
                return 0

        return None
    
    def delta(self, bump=0.01):

        S_up = self.S + bump
        S_down = self.S - bump

        price_up = self.mc_up.price_barrier(S_up, self.K, self.T, self.r, self.sigma, self.q,
                                        self.option_type, self.barrier_type, self.barrier_level)
        price_down = self.mc_down.price_barrier(S_down, self.K, self.T, self.r, self.sigma, self.q,
                                            self.option_type, self.barrier_type, self.barrier_level)

        return (price_up - price_down) / (2 * bump)

    def gamma(self, bump=0.01):

        S_up = self.S + bump
        S_down = self.S - bump

        price_up = self.mc_up.price_barrier(S_up, self.K, self.T, self.r, self.sigma, self.q,
                                        self.option_type, self.barrier_type, self.barrier_level)
        price_center = self.mc_center.price_barrier(self.S, self.K, self.T, self.r, self.sigma, self.q,
                                                self.option_type, self.barrier_type, self.barrier_level)
        price_down = self.mc_down.price_barrier(S_down, self.K, self.T, self.r, self.sigma, self.q,
                                            self.option_type, self.barrier_type, self.barrier_level)

        return (price_up - 2 * price_center + price_down) / (bump ** 2)

    def vega(self, bump=0.01):

        sigma_up = self.sigma + bump
        sigma_down = self.sigma - bump

        price_up = self.mc_up.price_barrier(self.S, self.K, self.T, self.r, sigma_up, self.q,
                                        self.option_type, self.barrier_type, self.barrier_level)
        price_down = self.mc_down.price_barrier(self.S, self.K, self.T, self.r, sigma_down, self.q,
                                            self.option_type, self.barrier_type, self.barrier_level)

        return (price_up - price_down) / (2 * bump) / 100

    def theta(self, bump=1/365):

        T_down = max(self.T - bump, 0)

        price_center = self.mc_center.price_barrier(self.S, self.K, self.T, self.r, self.sigma, self.q,
                                                self.option_type, self.barrier_type, self.barrier_level)
        price_down = self.mc_down.price_barrier(self.S, self.K, T_down, self.r, self.sigma, self.q,
                                            self.option_type, self.barrier_type, self.barrier_level)

        return (price_down - price_center) / bump

    def rho(self, bump=0.01):

        r_up = self.r + bump
        r_down = self.r - bump

        price_up = self.mc_up.price_barrier(self.S, self.K, self.T, r_up, self.sigma, self.q,
                                        self.option_type, self.barrier_type, self.barrier_level)
        price_down = self.mc_down.price_barrier(self.S, self.K, self.T, r_down, self.sigma, self.q,
                                            self.option_type, self.barrier_type, self.barrier_level)

        return (price_up - price_down) / (2 * bump) / 100

    def get_all_greeks(self):

        return {
            'delta': self.delta(),
            'gamma': self.gamma(),
            'vega': self.vega(),
            'theta': self.theta(),
            'rho': self.rho()
        }