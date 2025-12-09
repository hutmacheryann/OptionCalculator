# Option Calculator

A comprehensive Python-based option pricing tool supporting European, American, Asian, and Barrier options with Black-Scholes and Monte Carlo pricing methods.

## Features

- **Multiple Option Types:**
  - European options (analytical Black-Scholes pricing)
  - American options (Longstaff-Schwartz Monte Carlo)
  - Asian options (arithmetic/geometric averaging)
  - Barrier options (knock-in/knock-out)

- **Greeks Calculation:**
  - Delta (∂V/∂S)
  - Gamma (∂²V/∂S²)
  - Vega (∂V/∂σ)
  - Theta (∂V/∂T)
  - Rho (∂V/∂r)

- **Interactive Dashboard:**
  - Streamlit web interface with real-time calculations
  - Payoff diagrams and P&L visualization
  - Sensitivity analysis (price and delta vs spot)
  - Interactive parameter controls

## Installation

1. **Clone the repository:**
```bash
  git clone https://github.com/hutmacheryann/OptionCalculator.git
  cd OptionCalculator
```

2. **Install dependencies:**
```bash
  pip install numpy scipy streamlit plotly
```

## File Structure

```
OptionCalculator/
├── config/                              # Configuration files for different option types
│   ├── american_call.json
│   ├── american_put.json
│   ├── european_call.json
│   ├── european_put.json
│   ├── test_american_call_itm.json
│   ├── test_asian_arithmetic_call.json
│   ├── test_asian_geometric_call.json
│   ├── test_barrier_down_out_call.json
│   └── test_barrier_up_out_call.json
├── logic/                               # Core pricing logic
│   ├── __init__.py
│   ├── american.py                      # American option (LSM)
│   ├── asian.py                         # Asian option (MC)
│   ├── barrier.py                       # Barrier option (MC)
│   ├── black_scholes.py                 # Black-Scholes model
│   ├── european.py                      # European option (analytical)
│   ├── monte_carlo.py                   # Monte Carlo engine
│   └── option.py                        # Base option class
├── utils/                               # Utilities
│   ├── __init__.py
│   ├── validators.py                    # Parameter validation
│   └── io_handler.py                    # Config I/O and result formatting
├── calculator.py                        # Main calculator class
├── main.py                              # CLI interface
├── app.py                               # Streamlit dashboard
├── run_all_tests.sh                     # Test suite runner
└── README.md
```

## Usage

### Streamlit Dashboard (Optional)

Launch the interactive web interface:

```bash
  streamlit run app.py
```

Features:
- Real-time option pricing with parameter sliders
- Visual payoff diagrams
- Greeks calculation and display
- Sensitivity analysis charts
- Support for all option types

### Command Line Interface

Run calculations via CLI:

```bash
  python main.py --config config/european_call.json
```

#### CLI Options

```
python main.py --config <config_file> [options]

Required:
  --config, -c     Path to configuration JSON file

Optional:
  --output, -o     Path to output file (default: console)
  --format, -f     Output format: json or txt (default: json)
  --no-greeks      Skip Greeks calculation (faster)
  --simple         Minimal output (price only)
```

#### CLI Examples

1. **Calculate European call (console output):**
```bash
  python main.py --config config/european_call.json
```

2. **Calculate American put and save to JSON:**
```bash
  python main.py --config config/american_put.json --output results.json
```

3. **Calculate Asian option and save as text:**
```bash
  python main.py --config config/test_asian_arithmetic_call.json --output results.txt --format txt
```

4. **Quick price without Greeks:**
```bash
  python main.py --config config/test_barrier_down_out_call.json --no-greeks --simple
```

5. **Run all test cases:**
```bash
  bash run_all_tests.sh
```

## Configuration File Format

Configuration files use JSON format. Example for a European call:

```json
{
  "option_style": "european",
  "option_type": "call",
  "underlying_price": 100.0,
  "strike_price": 105.0,
  "time_to_maturity": 0.5,
  "volatility": 0.2,
  "risk_free_rate": 0.05,
  "dividend_yield": 0.02
}
```

### Required Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `option_style` | Option type: `european`, `american`, `asian`, `barrier` | `"european"` |
| `option_type` | Call or put | `"call"` |
| `underlying_price` | Current spot price (S) | `100.0` |
| `strike_price` | Strike price (K) | `105.0` |
| `time_to_maturity` | Time to expiration in years (T) | `0.5` |
| `volatility` | Volatility (σ) | `0.2` |
| `risk_free_rate` | Risk-free rate (r) | `0.05` |

### Optional Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `dividend_yield` | Continuous dividend yield (q) | `0` |
| `num_simulations` | Number of Monte Carlo paths | `10000` |
| `num_steps` | Time steps per simulation | `252` |
| `seed` | Random seed for reproducibility | `42` |

### Option-Specific Parameters

**Asian Options:**
```json
{
  "option_style": "asian",
  "average_type": "arithmetic" 
}
```

**Barrier Options:**
```json
{
  "option_style": "barrier",
  "barrier_type": "down-and-out",  
  "barrier_level": 90.0
}
```

## Pricing Methods

### European Options
- **Method:** Closed-form Black-Scholes formula
- **Greeks:** Analytical derivatives
- **Speed:** Instant (no simulation required)

### American Options
- **Method:** Longstaff-Schwartz Monte Carlo (LSM)
- **Greeks:** Finite differences with Common Random Numbers (CRN)
- **Speed:** Moderate (depends on `num_simulations`)

### Asian Options
- **Method:** Monte Carlo simulation
- **Averaging:** Arithmetic or geometric mean of path
- **Greeks:** Finite differences with CRN

### Barrier Options
- **Method:** Monte Carlo simulation
- **Types:** Up/down, knock-in/knock-out
- **Greeks:** Finite differences with CRN

## Technical Details

### Monte Carlo Engine
- **Path Generation:** Geometric Brownian Motion (GBM)
- **Variance Reduction:** Common Random Numbers (CRN) for Greeks
- **Reproducibility:** Seeded RNG (default seed: 42)

### Greeks Calculation
- **European:** Analytical formulas
- **Other Options:** Finite differences with controlled bumps
  - Delta bump: 0.5
  - Gamma bump: 5.0 (larger for LSM stability)
  - Vega bump: 0.01
  - Rho bump: 0.01
  - Theta bump: 1/365

### Validation
- Automatic parameter validation (positive prices, valid ranges)
- Barrier consistency checks (up barriers > spot, down barriers < spot)
- Average type validation for Asian options

## Example Output

```
============================================================
RESULTS
============================================================

Input Parameters:
------------------------------------------------------------
  Option Style: european
  Option Type: call
  Underlying Price: 100.0
  Strike Price: 105.0
  Time To Maturity: 0.5
  Volatility: 0.2
  Risk Free Rate: 0.05
  Dividend Yield: 0.02

------------------------------------------------------------
Option Price: $3.7041
------------------------------------------------------------

Greeks:
------------------------------------------------------------
  Delta:   0.441538
  Gamma:   0.023456
  Vega:    18.234567
  Theta:   -0.015234
  Rho:     19.876543
============================================================
```

## License

This project is available for educational and research purposes.

## Dependencies

- **numpy:** Numerical computations and array operations
- **scipy:** Statistical distributions (normal CDF/PDF)
- **streamlit:** Interactive web dashboard (optional)
- **plotly:** Interactive charts (optional, for dashboard)

## Future Enhancements

- Implied volatility calculation
- Option Greeks surface visualization
- Additional exotic option types (lookback, chooser)
- Historical volatility estimation
- Real-time market data integration

## Contact

For questions or suggestions, please open an issue on GitHub.