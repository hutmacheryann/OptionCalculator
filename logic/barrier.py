import numpy as np
from .monte_carlo import MonteCarloEngine
from scipy.stats import norm


class BarrierOption:
    '''
    Barrier option priced via Monte Carlo simulation, with an optional 
    closed-form formula for specific cases.

    Attributes: 
    S : float
        Current underlying price
    K : float
        Strike price
    T : float
        Time to maturity (years)
    r : float
        Risk-free rate
    sigma : float
        Volatility of the underlying
    q : float, optional
        Dividend yield (default: 0)
    option_type : str, optional
        "call" or "put" (default: "call")
    barrier_type : str, optional
        Barrier type: "up-and-out", "up-and-in", "down-and-out", or "down-and-in"
        (default: "down-and-out")
    barrier_level : float, optional
        Barrier level H. Required for barrier options
    num_simulations : int, optional
        Number of Monte Carlo paths (default: 10000)
    num_steps : int, optional
        Number of time steps per path (default: 252)
    '''

    def __init__(self, S, K, T, r, sigma, q=0, option_type='call', barrier_type='down-and-out', barrier_level=None, num_simulations=10000, num_steps=252):
        
        # Store option parameters
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
        self.option_type = option_type.lower()
        self.barrier_type = barrier_type.lower()
        self.barrier_level = barrier_level

        # Monte Carlo engine parameters
        self.num_simulations = num_simulations
        self.num_steps = num_steps

        # Initialise Monte Carlo engine
        self.mc_engine = MonteCarloEngine(num_simulations, num_steps)

        if barrier_level is None:
            raise ValueError("barrier_level is required for barrier options")

    def price(self):
        '''Monte Carlo price of the barrier option'''
        return self.mc_engine.price_barrier(
            self.S, self.K, self.T, self.r, self.sigma, self.q,
            self.option_type, self.barrier_type, self.barrier_level
        )

    def price_closed_form(self):
        '''Closed-form price for a down-and-out call (if applicable)'''

        # Closed-form implemented only for down-and-out call
        if self.barrier_type == 'down-and-out' and self.option_type == 'call':
            H = self.barrier_level
            # If the spot is already below the barrier, option is knocked out
            if self.S <= H:
                return 0

            # Auxiliary parameters for the closed-form formula
            lambda_val = (self.r - self.q + 0.5 * self.sigma**2) / (self.sigma**2)
            y = np.log(H**2 / (self.S * self.K)) / (self.sigma * np.sqrt(self.T)) + lambda_val * self.sigma * np.sqrt(self.T)
            x1 = np.log(self.S / H) / (self.sigma * np.sqrt(self.T)) + lambda_val * self.sigma * np.sqrt(self.T)
            y1 = np.log(H / self.S) / (self.sigma * np.sqrt(self.T)) + lambda_val * self.sigma * np.sqrt(self.T)

            # Only valid when strike is above the barrier
            if self.K > H:
                from .black_scholes import BlackScholesModel
                bs = BlackScholesModel()

                # Vanilla European call price
                vanilla_call = bs.call_price(self.S, self.K, self.T, self.r, self.sigma, self.q)

                # Barrier correction term (subtracted from vanilla call)
                correction = (self.S * np.exp(-self.q * self.T) * (H / self.S)**(2 * lambda_val) * norm.cdf(y) -
                             self.K * np.exp(-self.r * self.T) * (H / self.S)**(2 * lambda_val - 2) * norm.cdf(y - self.sigma * np.sqrt(self.T)))

                return vanilla_call - correction
            else:
                # If K <= H, this closed-form expression is not applicable
                return 0
        # If closed-form is not implemented for this configuration
        return None

    def delta(self, bump=0.01):
        """Sensitivity of option value to underlying price (∂V/∂S)"""
        S_up = self.S + bump
        S_down = self.S - bump

        mc_up = MonteCarloEngine(self.num_simulations, self.num_steps, seed=42)
        mc_down = MonteCarloEngine(self.num_simulations, self.num_steps, seed=42)

        price_up = mc_up.price_barrier(S_up, self.K, self.T, self.r, self.sigma, self.q,
                                        self.option_type, self.barrier_type, self.barrier_level)
        price_down = mc_down.price_barrier(S_down, self.K, self.T, self.r, self.sigma, self.q,
                                            self.option_type, self.barrier_type, self.barrier_level)

        return (price_up - price_down) / (2 * bump)

    def gamma(self, bump=0.01):
        """Curvature of option value w.r.t underlying price (∂²V/∂S²)"""
        S_up = self.S + bump
        S_down = self.S - bump

        mc_up = MonteCarloEngine(self.num_simulations, self.num_steps, seed=42)
        mc_center = MonteCarloEngine(self.num_simulations, self.num_steps, seed=42)
        mc_down = MonteCarloEngine(self.num_simulations, self.num_steps, seed=42)

        price_up = mc_up.price_barrier(S_up, self.K, self.T, self.r, self.sigma, self.q,
                                        self.option_type, self.barrier_type, self.barrier_level)
        price_center = mc_center.price_barrier(self.S, self.K, self.T, self.r, self.sigma, self.q,
                                                self.option_type, self.barrier_type, self.barrier_level)
        price_down = mc_down.price_barrier(S_down, self.K, self.T, self.r, self.sigma, self.q,
                                            self.option_type, self.barrier_type, self.barrier_level)

        return (price_up - 2 * price_center + price_down) / (bump ** 2)

    def vega(self, bump=0.01):
        """Sensitivity to volatility (∂V/∂σ), per 1% change"""
        sigma_up = self.sigma + bump
        sigma_down = self.sigma - bump

        mc_up = MonteCarloEngine(self.num_simulations, self.num_steps, seed=42)
        mc_down = MonteCarloEngine(self.num_simulations, self.num_steps, seed=42)

        price_up = mc_up.price_barrier(self.S, self.K, self.T, self.r, sigma_up, self.q,
                                        self.option_type, self.barrier_type, self.barrier_level)
        price_down = mc_down.price_barrier(self.S, self.K, self.T, self.r, sigma_down, self.q,
                                            self.option_type, self.barrier_type, self.barrier_level)

        return (price_up - price_down) / (2 * bump) / 100

    def theta(self, bump=1/365):
        """Time decay of the option value (∂V/∂t), per day"""
        T_down = max(self.T - bump, 0)

        mc_center = MonteCarloEngine(self.num_simulations, self.num_steps, seed=42)
        mc_down = MonteCarloEngine(self.num_simulations, self.num_steps, seed=42)

        price_center = mc_center.price_barrier(self.S, self.K, self.T, self.r, self.sigma, self.q,
                                                self.option_type, self.barrier_type, self.barrier_level)
        price_down = mc_down.price_barrier(self.S, self.K, T_down, self.r, self.sigma, self.q,
                                            self.option_type, self.barrier_type, self.barrier_level)

        return (price_down - price_center) / bump

    def rho(self, bump=0.01):
        """Sensitivity to interest rate (∂V/∂r), per 1% change"""
        r_up = self.r + bump
        r_down = self.r - bump

        mc_up = MonteCarloEngine(self.num_simulations, self.num_steps, seed=42)
        mc_down = MonteCarloEngine(self.num_simulations, self.num_steps, seed=42)

        price_up = mc_up.price_barrier(self.S, self.K, self.T, r_up, self.sigma, self.q,
                                        self.option_type, self.barrier_type, self.barrier_level)
        price_down = mc_down.price_barrier(self.S, self.K, self.T, r_down, self.sigma, self.q,
                                            self.option_type, self.barrier_type, self.barrier_level)

        return (price_up - price_down) / (2 * bump) / 100

    def get_all_greeks(self):
        """Compute all Greeks and return as dictionary"""
        return {
            'delta': self.delta(),
            'gamma': self.gamma(),
            'vega': self.vega(),
            'theta': self.theta(),
            'rho': self.rho()
        }