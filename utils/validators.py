"""
Parameter validation utilities.

Validates option parameters to ensure they're physically meaningful.
"""


def validate_option_params(S, K, T, r, sigma, q=0):
    """
    Validate basic option parameters.

    Parameters
    ----------
    S : float
        Spot price (must be positive)
    K : float
        Strike price (must be positive)
    T : float
        Time to maturity (must be non-negative)
    r : float
        Risk-free rate
    sigma : float
        Volatility (must be positive)
    q : float, optional
        Dividend yield (must be non-negative)

    Returns
    -------
    is_valid : bool
        True if all parameters are valid
    error_msg : str or None
        Error message if invalid, None otherwise
    """
    errors = []

    # Prices need to be positive
    if S <= 0:
        errors.append("Underlying price must be positive")

    if K <= 0:
        errors.append("Strike price must be positive")

    # Can't have negative time left
    if T < 0:
        errors.append("TTM must be positive")

    # Volatility has to be positive
    if sigma <= 0:
        errors.append("Volatility must be positive")

    # Dividend yield should be non-negative
    if q < 0:
        errors.append("Dividend yield must be positive")

    # If we found any problems, return them
    if errors:
        return False, "; ".join(errors)

    return True, None


def validate_barrier_params(barrier_type, barrier_level, S):
    """
    Validate barrier option parameters.

    Ensures barrier level is positioned correctly relative to spot price.

    Parameters
    ----------
    barrier_type : str
        One of: 'up-and-out', 'up-and-in', 'down-and-out', 'down-and-in'
    barrier_level : float
        Barrier price level (must be positive)
    S : float
        Current spot price

    Returns
    -------
    is_valid : bool
        True if parameters are valid
    error_msg : str or None
        Error message if invalid, None otherwise
    """
    valid_types = ['up-and-out', 'up-and-in', 'down-and-out', 'down-and-in']

    # Make sure barrier type is one we know about
    if barrier_type.lower() not in valid_types:
        return False, f"Invalid barrier_type. Must be one of: {', '.join(valid_types)}"

    # Barrier level has to be positive
    if barrier_level <= 0:
        return False, "Barrier level must be positive"

    # Up barriers should be above the current price (otherwise they're useless)
    if 'up' in barrier_type.lower() and barrier_level <= S:
        return False, "For up-barriers, barrier level must be > stock price"

    # Down barriers should be below the current price
    if 'down' in barrier_type.lower() and barrier_level >= S:
        return False, "For down-barriers, barrier level must be < stock price"

    return True, None


def validate_asian_params(average_type):
    """
    Validate Asian option parameters.

    Parameters
    ----------
    average_type : str
        Must be 'arithmetic' or 'geometric'

    Returns
    -------
    is_valid : bool
        True if parameter is valid
    error_msg : str or None
        Error message if invalid, None otherwise
    """
    valid_types = ['arithmetic', 'geometric']

    if average_type.lower() not in valid_types:
        return False, f"Invalid average_type. Must be one of: {', '.join(valid_types)}"

    return True, None