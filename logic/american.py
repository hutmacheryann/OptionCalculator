from .option import Option


class AmericanOption(Option):

    def price(self):
        return self.mc_engine.price_american(self.S, self.K, self.T, self.r, self.sigma, self.q, self.option_type)
    
    def delta(self, bump=0.01):

        S_up = self.S + bump
        S_down = self.S - bump

        price_up = self.mc_up.price_american(S_up, self.K, self.T, self.r, self.sigma, self.q, self.option_type)
        price_down = self.mc_down.price_american(S_down, self.K, self.T, self.r, self.sigma, self.q, self.option_type)

        return (price_up - price_down) / (2 * bump)

    def gamma(self, bump=0.01):

        S_up = self.S + bump
        S_down = self.S - bump

        price_up = self.mc_up.price_american(S_up, self.K, self.T, self.r, self.sigma, self.q, self.option_type)
        price_center = self.mc_center.price_american(self.S, self.K, self.T, self.r, self.sigma, self.q, self.option_type)
        price_down = self.mc_down.price_american(S_down, self.K, self.T, self.r, self.sigma, self.q, self.option_type)

        return (price_up - 2 * price_center + price_down) / (bump ** 2)

    def vega(self, bump=0.01):

        sigma_up = self.sigma + bump
        sigma_down = self.sigma - bump

        price_up = self.mc_up.price_american(self.S, self.K, self.T, self.r, sigma_up, self.q, self.option_type)
        price_down = self.mc_down.price_american(self.S, self.K, self.T, self.r, sigma_down, self.q, self.option_type)

        return (price_up - price_down) / (2 * bump) / 100

    def theta(self, bump=1/365):

        T_down = max(self.T - bump, 0)

        price_center = self.mc_center.price_american(self.S, self.K, self.T, self.r, self.sigma, self.q, self.option_type)
        price_down = self.mc_down.price_american(self.S, self.K, T_down, self.r, self.sigma, self.q, self.option_type)

        return (price_down - price_center) / bump

    def rho(self, bump=0.01):

        r_up = self.r + bump
        r_down = self.r - bump

        price_up = self.mc_up.price_american(self.S, self.K, self.T, r_up, self.sigma, self.q, self.option_type)
        price_down = self.mc_down.price_american(self.S, self.K, self.T, r_down, self.sigma, self.q, self.option_type)

        return (price_up - price_down) / (2 * bump) / 100
    
    def get_all_greeks(self):

        return {
            'delta': self.delta(),
            'gamma': self.gamma(),
            'vega': self.vega(),
            'theta': self.theta(),
            'rho': self.rho()
        }