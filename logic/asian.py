# Import necessary tools for Monte Carlo simulations
from .monte_carlo import MonteCarloEngine

class AsianOption:

    '''
    Represents an Asian option priced via Monte Carlo simulation.

    Parameters
    ----------
    S : float
        Current underlying price.
    K : float
        Strike price.
    T : float
        Time to maturity (in years).
    r : float
        Risk-free interest rate.
    sigma : float
        Volatility of the underlying.
    q : float, optional
        Dividend yield (default: 0).
    option_type : str, optional
        "call" or "put" (default: "call").
    average_type : str, optional
        "arithmetic" or "geometric" averaging (default: "arithmetic").
    num_simulations : int, optional
        Number of Monte Carlo paths (default: 10,000).
    num_steps : int, optional
        Number of time steps per path (default: 252).
    '''

    def __init__(self, S, K, T, r, sigma, q=0, option_type='call', average_type='arithmetic', num_simulations=10000, num_steps=252):
        # Store option parameters
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
        self.option_type = option_type.lower()
        self.average_type = average_type.lower()

        # Monte Carlo engine parameters
        self.num_simulations = num_simulations
        self.num_steps = num_steps
        # Initialise Monte Carlo engine using the imported class
        self.mc_engine = MonteCarloEngine(num_simulations, num_steps)

    def price(self): # Simple wrapper for pricing function
        '''
        Computes the price of an Asian option using the Monte Carlo engine (initialised earlier).
        Returns float : estimated option price.
        '''
        return self.mc_engine.price_asian(
            self.S, self.K, self.T, self.r, self.sigma,
            self.q, self.option_type, self.average_type
        )
    
    '''
    -----------------------------------------------------------
    Greeks (finite differences)
    Each Greek re-runs Monte Carlo with a bumped parameter
    Using a fixed seed ensures path consistency (reduces variance)
    Using central finite differences for better accuracy
    --------------------------------------------------------------
    '''
    def delta(self, bump=0.01):
        '''
        Delta = dPrice/dS: sensitivity to underlying price.
        '''
        # Bumped underlying prices
        S_up = self.S + bump 
        S_down = self.S - bump

        # Initialise two Monte Carlo engines for up and down scenarios
        mc_up = MonteCarloEngine(self.num_simulations, self.num_steps, seed=42)
        mc_down = MonteCarloEngine(self.num_simulations, self.num_steps, seed=42)

        # Price options for bumped underlying prices
        price_up = mc_up.price_asian(S_up, self.K, self.T, self.r, self.sigma, self.q,
                                      self.option_type, self.average_type)
        price_down = mc_down.price_asian(S_down, self.K, self.T, self.r, self.sigma, self.q,
                                          self.option_type, self.average_type)
        
        # Approximate delta using central finite difference
        return (price_up - price_down) / (2 * bump)

    def gamma(self, bump=0.01):
        '''
        Gamma = d²Price/dS²: sensitivity of delta to underlying price.
        '''
        # Bumped underlying prices
        S_up = self.S + bump
        S_down = self.S - bump

        # Initialise three Monte Carlo engines for up, center, and down scenarios
        mc_up = MonteCarloEngine(self.num_simulations, self.num_steps, seed=42)
        mc_center = MonteCarloEngine(self.num_simulations, self.num_steps, seed=42)
        mc_down = MonteCarloEngine(self.num_simulations, self.num_steps, seed=42)

        # Price options for bumped and original underlying prices
        price_up = mc_up.price_asian(S_up, self.K, self.T, self.r, self.sigma, self.q,
                                      self.option_type, self.average_type)
        price_center = mc_center.price_asian(self.S, self.K, self.T, self.r, self.sigma, self.q,
                                              self.option_type, self.average_type)
        price_down = mc_down.price_asian(S_down, self.K, self.T, self.r, self.sigma, self.q,
                                          self.option_type, self.average_type)

        # Approximate gamma using central finite difference
        return (price_up - 2 * price_center + price_down) / (bump ** 2)

    def vega(self, bump=0.01):
        '''
        Vega = dPrice/dSigma: sensitivity to volatility.
        '''

        # Bumped volatilities
        sigma_up = self.sigma + bump
        sigma_down = self.sigma - bump

        # Initialise two Monte Carlo engines for up and down scenarios
        mc_up = MonteCarloEngine(self.num_simulations, self.num_steps, seed=42)
        mc_down = MonteCarloEngine(self.num_simulations, self.num_steps, seed=42)

        # Price options for bumped volatilities
        price_up = mc_up.price_asian(self.S, self.K, self.T, self.r, sigma_up, self.q,
                                      self.option_type, self.average_type)
        price_down = mc_down.price_asian(self.S, self.K, self.T, self.r, sigma_down, self.q,
                                          self.option_type, self.average_type)

        # Approximate vega using central finite difference (divide by 100 to express per 1% change in volatility)
        return (price_up - price_down) / (2 * bump) / 100

    def theta(self, bump=1/365):
        '''
        Theta = dPrice/dT: sensitivity to time to maturity.
        '''
        # Bumped time to maturity (ensure non-negative)
        T_down = max(self.T - bump, 0)

        # Initialise two Monte Carlo engines for center and down scenarios
        mc_center = MonteCarloEngine(self.num_simulations, self.num_steps, seed=42)
        mc_down = MonteCarloEngine(self.num_simulations, self.num_steps, seed=42)

        # Price options for original and bumped time to maturity
        price_center = mc_center.price_asian(self.S, self.K, self.T, self.r, self.sigma, self.q,
                                              self.option_type, self.average_type)
        price_down = mc_down.price_asian(self.S, self.K, T_down, self.r, self.sigma, self.q,
                                          self.option_type, self.average_type)

        # Approximate theta using finite difference (negative sign to reflect decrease in time)
        return (price_down - price_center) / bump

    def rho(self, bump=0.01):
        '''
        Rho = dPrice/dr: sensitivity to risk-free interest rate.
        '''
        # Bumped interest rates
        r_up = self.r + bump
        r_down = self.r - bump

        # Initialise two Monte Carlo engines for up and down scenarios
        mc_up = MonteCarloEngine(self.num_simulations, self.num_steps, seed=42)
        mc_down = MonteCarloEngine(self.num_simulations, self.num_steps, seed=42)

        # Price options for bumped interest rates
        price_up = mc_up.price_asian(self.S, self.K, self.T, r_up, self.sigma, self.q,
                                      self.option_type, self.average_type)
        price_down = mc_down.price_asian(self.S, self.K, self.T, r_down, self.sigma, self.q,
                                          self.option_type, self.average_type)

        # Approximate rho using central finite difference (divide by 100 to express per 1% change in rate)
        return (price_up - price_down) / (2 * bump) / 100

    def get_all_greeks(self):
        '''
        Computes all Greeks and returns them in a dictionary.
        '''
        return {
            'delta': self.delta(),
            'gamma': self.gamma(),
            'vega': self.vega(),
            'theta': self.theta(),
            'rho': self.rho()
        }
