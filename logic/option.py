from .monte_carlo import MonteCarloEngine

class Option:

    def __init__(self, S, K, T, r, sigma, q=0, option_type='call', num_simulations=10000, num_steps=252):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
        self.option_type = option_type.lower()
        self.num_simulations = num_simulations
        self.num_steps = num_steps
        self.mc_engine = MonteCarloEngine(num_simulations, num_steps)
        self.mc_up = MonteCarloEngine(self.num_simulations, self.num_steps, seed=42)
        self.mc_center = MonteCarloEngine(self.num_simulations, self.num_steps, seed=42)
        self.mc_down = MonteCarloEngine(self.num_simulations, self.num_steps, seed=42)