from logic.option import EuropeanOption
from logic.option import AmericanOption
from logic.option import AsianOption
from logic.option import BarrierOption
from utils.validators import (validate_option_params, validate_barrier_params, validate_asian_params)


class OptionCalculator:

    def __init__(self, config):
        self.config = config
        self.option = None
        self.results = {}

    def create_option(self):

        S = float(self.config['underlying_price'])
        K = float(self.config['strike_price'])
        T = float(self.config['time_to_maturity'])
        r = float(self.config['risk_free_rate'])
        sigma = float(self.config['volatility'])
        q = float(self.config.get('dividend_yield', 0))
        option_type = self.config['option_type'].lower()
        option_style = self.config['option_style'].lower()

        # throw error if failed
        is_valid, error_msg = validate_option_params(S, K, T, r, sigma, q)
        if not is_valid:
            raise ValueError(f"Invalid parameters: {error_msg}")

        # Monte Carlo
        num_simulations = int(self.config.get('num_simulations', 10000))
        num_steps = int(self.config.get('num_steps', 252))

        # create the correct option stats
        if option_style == 'european':
            self.option = EuropeanOption(S, K, T, r, sigma, q, option_type)

        elif option_style == 'american':
            self.option = AmericanOption(S, K, T, r, sigma, q, option_type,
                                        num_simulations, num_steps)

        elif option_style == 'asian':
            average_type = self.config.get('average_type', 'arithmetic')
            is_valid, error_msg = validate_asian_params(average_type)
            if not is_valid:
                raise ValueError(f"Invalid Asian option parameters: {error_msg}")

            self.option = AsianOption(S, K, T, r, sigma, q, option_type,
                                     average_type, num_simulations, num_steps)

        elif option_style == 'barrier':
            barrier_type = self.config['barrier_type'].lower()
            barrier_level = float(self.config['barrier_level'])

            is_valid, error_msg = validate_barrier_params(barrier_type, barrier_level, S)
            if not is_valid:
                raise ValueError(f"Invalid barrier option parameters: {error_msg}")

            self.option = BarrierOption(S, K, T, r, sigma, q, option_type,
                                       barrier_type, barrier_level,
                                       num_simulations, num_steps)

        else:
            raise ValueError(f"Invalid option style: {option_style}. "
                           f"Must be one of: european, american, asian, barrier")

        return self.option

    def calculate(self, compute_greeks=True):

        if self.option is None:
            self.create_option()

        greeks = None
        if compute_greeks:
            greeks = self.option.get_all_greeks()

        self.results = {
            'price': self.option.price(),
            'greeks': greeks,
            'parameters': self.config
        }

        return self.results

    def get_results(self):
        return self.results


def calculate_from_config(config, compute_greeks=True):
    calculator = OptionCalculator(config)
    return calculator.calculate(compute_greeks)