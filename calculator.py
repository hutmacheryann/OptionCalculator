"""
Main calculator class for option pricing.

Orchestrates option creation and calculation based on config parameters.
"""
from logic.european import EuropeanOption
from logic.american import AmericanOption
from logic.asian import AsianOption
from logic.barrier import BarrierOption
from utils.validators import validate_option_params, validate_barrier_params, validate_asian_params


class OptionCalculator:
    """
        Main calculator that creates and prices options based on config.

        Parameters
        ----------
        config : dict
            Configuration dictionary containing option parameters:
            - option_style: 'european', 'american', 'asian', or 'barrier'
            - option_type: 'call' or 'put'
            - underlying_price, strike_price, time_to_maturity, etc.

    """

    def __init__(self, config):
        self.config = config
        self.option = None
        self.results = {}

    def create_option(self):
        """
        Create the appropriate option object based on config.

        Returns
        -------
        Option
            Instance of appropriate option subclass

        Raises
        ------
        ValueError
            If parameters are invalid or option style is unsupported
        """
        S = float(self.config['underlying_price'])
        K = float(self.config['strike_price'])
        T = float(self.config['time_to_maturity'])
        r = float(self.config['risk_free_rate'])
        sigma = float(self.config['volatility'])
        q = float(self.config.get('dividend_yield', 0))
        option_type = self.config['option_type'].lower()
        option_style = self.config['option_style'].lower()

        is_valid, error_msg = validate_option_params(S, K, T, r, sigma, q)
        if not is_valid:
            raise ValueError(f"Invalid parameters: {error_msg}")

        num_simulations = int(self.config.get('num_simulations', 10000))
        num_steps = int(self.config.get('num_steps', 252))
        seed = int(self.config.get('seed', 42))

        if option_style == 'european':
            self.option = EuropeanOption(S, K, T, r, sigma, q, option_type)

        elif option_style == 'american':
            self.option = AmericanOption(S, K, T, r, sigma, q, option_type,
                                         num_simulations, num_steps, seed)

        elif option_style == 'asian':
            average_type = self.config.get('average_type', 'arithmetic')
            is_valid, error_msg = validate_asian_params(average_type)
            if not is_valid:
                raise ValueError(f"Invalid Asian option parameters: {error_msg}")
            self.option = AsianOption(S, K, T, r, sigma, q, option_type,
                                      average_type, num_simulations, num_steps, seed)

        elif option_style == 'barrier':
            barrier_type = self.config['barrier_type'].lower()
            barrier_level = float(self.config['barrier_level'])
            is_valid, error_msg = validate_barrier_params(barrier_type, barrier_level, S)
            if not is_valid:
                raise ValueError(f"Invalid barrier option parameters: {error_msg}")
            self.option = BarrierOption(S, K, T, r, sigma, q, option_type,
                                        barrier_type, barrier_level,
                                        num_simulations, num_steps, seed)
        else:
            raise ValueError(f"Invalid option style: {option_style}")

        return self.option

    def calculate(self, compute_greeks=True):
        """
        Run the full calculation (price and optionally Greeks).

        Parameters
        ----------
        compute_greeks : bool, optional
            Whether to calculate Greeks (default: True)

        Returns
        -------
        dict
            Results dictionary with keys:
            - 'price': float
            - 'greeks': dict or None
            - 'parameters': dict
        """
        if self.option is None:
            self.create_option()

        # Calculate price first
        price = self.option.price()

        # Then calculate Greeks (each Greek resets RNG for reproducibility)
        greeks = self.option.get_all_greeks() if compute_greeks else None

        self.results = {
            'price': price,
            'greeks': greeks,
            'parameters': self.config
        }
        return self.results

    def get_results(self):
        """Return stored results from last calculation."""
        return self.results


def calculate_from_config(config, compute_greeks=True):
    """
    Convenience function for quick calculations.

    Parameters
    ----------
    config : dict
        Option configuration
    compute_greeks : bool, optional
        Whether to calculate Greeks (default: True)

    Returns
    -------
    dict
        Calculation results
    """
    calculator = OptionCalculator(config)
    return calculator.calculate(compute_greeks)