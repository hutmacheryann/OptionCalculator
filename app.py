"""
Option Calculator - Streamlit Dashboard
Run with: streamlit run app.py
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from calculator import OptionCalculator

# Page config
st.set_page_config(
    page_title="Option Calculator",
    page_icon="📈",
    layout="wide"
)

# Cache the calculation to avoid recomputing
@st.cache_data
def calculate_option(config_tuple, compute_greeks):
    """Cached option calculation."""
    config = dict(config_tuple)
    calculator = OptionCalculator(config)
    return calculator.calculate(compute_greeks=compute_greeks)

@st.cache_data
def calculate_sensitivity(base_config_tuple, param_name, param_range, compute_greeks=False):
    """Cached sensitivity calculation with reduced simulations."""
    results = []
    base_config = dict(base_config_tuple)

    # Use fewer simulations for sensitivity analysis
    sensitivity_config = base_config.copy()
    sensitivity_config["num_simulations"] = min(base_config.get("num_simulations", 10000), 10000)

    for val in param_range:
        temp_config = sensitivity_config.copy()
        temp_config[param_name] = val
        try:
            calc = OptionCalculator(temp_config)
            res = calc.calculate(compute_greeks=compute_greeks)
            results.append({
                'param': val,
                'price': res['price'],
                'greeks': res.get('greeks')
            })
        except:
            results.append({'param': val, 'price': np.nan, 'greeks': None})

    return results

# Title
st.title("📈 Option Pricing Calculator")
st.markdown("Price European, American, Asian, and Barrier options with Greeks calculation")

# Sidebar - Input Parameters
st.sidebar.header("Option Parameters")

# Option Style & Type
option_style = st.sidebar.selectbox(
    "Option Style",
    ["european", "american", "asian", "barrier"],
    format_func=lambda x: x.title()
)

option_type = st.sidebar.radio(
    "Option Type",
    ["call", "put"],
    format_func=lambda x: x.title(),
    horizontal=True
)

st.sidebar.divider()

# Market Parameters
st.sidebar.subheader("Market Parameters")

underlying_price = st.sidebar.number_input(
    "Underlying Price (S)",
    min_value=0.01,
    max_value=10000.0,
    value=100.0,
    step=1.0
)

strike_price = st.sidebar.number_input(
    "Strike Price (K)",
    min_value=0.01,
    max_value=10000.0,
    value=100.0,
    step=1.0
)

time_to_maturity = st.sidebar.number_input(
    "Time to Maturity (years)",
    min_value=0.01,
    max_value=10.0,
    value=1.0,
    step=0.1
)

volatility = st.sidebar.number_input(
    "Volatility (σ)",
    min_value=0.01,
    max_value=2.0,
    value=0.3,
    step=0.01
)

risk_free_rate = st.sidebar.number_input(
    "Risk-Free Rate (r)",
    min_value=0.0,
    max_value=0.5,
    value=0.05,
    step=0.01
)

dividend_yield = st.sidebar.number_input(
    "Dividend Yield (q)",
    min_value=0.0,
    max_value=0.5,
    value=0.0,
    step=0.01
)

st.sidebar.divider()

# Monte Carlo Settings
st.sidebar.subheader("Monte Carlo Settings")

num_simulations = st.sidebar.select_slider(
    "Number of Simulations",
    options=[1000, 5000, 10000, 50000, 100000],
    value=10000
)

num_steps = st.sidebar.select_slider(
    "Number of Steps",
    options=[50, 100, 252, 500],
    value=252
)

# Style-specific parameters
if option_style == "asian":
    st.sidebar.divider()
    st.sidebar.subheader("Asian Option Settings")
    average_type = st.sidebar.selectbox(
        "Average Type",
        ["arithmetic", "geometric"],
        format_func=lambda x: x.title()
    )

if option_style == "barrier":
    st.sidebar.divider()
    st.sidebar.subheader("Barrier Option Settings")
    barrier_type = st.sidebar.selectbox(
        "Barrier Type",
        ["down-and-out", "down-and-in", "up-and-out", "up-and-in"],
        format_func=lambda x: x.replace("-", " ").title()
    )

    if "down" in barrier_type:
        default_barrier = underlying_price * 0.9
    else:
        default_barrier = underlying_price * 1.1

    barrier_level = st.sidebar.number_input(
        "Barrier Level",
        min_value=0.01,
        max_value=10000.0,
        value=default_barrier,
        step=1.0
    )

# Build config dict
config = {
    "option_style": option_style,
    "option_type": option_type,
    "underlying_price": underlying_price,
    "strike_price": strike_price,
    "time_to_maturity": time_to_maturity,
    "volatility": volatility,
    "risk_free_rate": risk_free_rate,
    "dividend_yield": dividend_yield,
    "num_simulations": num_simulations,
    "num_steps": num_steps
}

if option_style == "asian":
    config["average_type"] = average_type

if option_style == "barrier":
    config["barrier_type"] = barrier_type
    config["barrier_level"] = barrier_level

# Convert to tuple for caching (dicts aren't hashable)
config_tuple = tuple(sorted(config.items()))

# Calculate button
st.sidebar.divider()
compute_greeks = st.sidebar.checkbox("Calculate Greeks", value=True)
calculate_button = st.sidebar.button("Calculate", type="primary", use_container_width=True)

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Results")

    if calculate_button:
        try:
            with st.spinner("Calculating..."):
                results = calculate_option(config_tuple, compute_greeks)

            st.metric(
                label="Option Price",
                value=f"${results['price']:.4f}"
            )

            if compute_greeks and results.get('greeks'):
                st.divider()
                st.markdown("**Greeks**")
                greeks = results['greeks']

                gcol1, gcol2 = st.columns(2)
                with gcol1:
                    st.metric("Delta (Δ)", f"{greeks['delta']:.4f}")
                    st.metric("Gamma (Γ)", f"{greeks['gamma']:.6f}")
                    st.metric("Vega (ν)", f"{greeks['vega']:.4f}")
                with gcol2:
                    st.metric("Theta (Θ)", f"{greeks['theta']:.4f}")
                    st.metric("Rho (ρ)", f"{greeks['rho']:.4f}")

            st.session_state['results'] = results
            st.session_state['config'] = config
            st.session_state['config_tuple'] = config_tuple

        except Exception as e:
            st.error(f"Error: {str(e)}")

    elif 'results' in st.session_state:
        results = st.session_state['results']

        st.metric(
            label="Option Price",
            value=f"${results['price']:.4f}"
        )

        if results.get('greeks'):
            st.divider()
            st.markdown("**Greeks**")
            greeks = results['greeks']

            gcol1, gcol2 = st.columns(2)
            with gcol1:
                st.metric("Delta (Δ)", f"{greeks['delta']:.4f}")
                st.metric("Gamma (Γ)", f"{greeks['gamma']:.6f}")
                st.metric("Vega (ν)", f"{greeks['vega']:.4f}")
            with gcol2:
                st.metric("Theta (Θ)", f"{greeks['theta']:.4f}")
                st.metric("Rho (ρ)", f"{greeks['rho']:.4f}")
    else:
        st.info("Configure parameters and click 'Calculate' to see results.")

with col2:
    st.subheader("Payoff Diagram")

    spot_range = np.linspace(underlying_price * 0.5, underlying_price * 1.5, 100)

    if option_type == "call":
        intrinsic = np.maximum(spot_range - strike_price, 0)
    else:
        intrinsic = np.maximum(strike_price - spot_range, 0)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=spot_range,
        y=intrinsic,
        mode='lines',
        name='Payoff at Expiration',
        line=dict(color='#1f77b4', width=2)
    ))

    if 'results' in st.session_state:
        price = st.session_state['results']['price']
        pnl = intrinsic - price
        fig.add_trace(go.Scatter(
            x=spot_range,
            y=pnl,
            mode='lines',
            name='P&L (Payoff - Premium)',
            line=dict(color='#2ca02c', width=2, dash='dash')
        ))

    fig.add_vline(x=strike_price, line_dash="dot", line_color="gray",
                  annotation_text=f"K={strike_price}")
    fig.add_vline(x=underlying_price, line_dash="dot", line_color="orange",
                  annotation_text=f"S={underlying_price}")

    if option_style == "barrier":
        fig.add_vline(x=barrier_level, line_dash="dash", line_color="red",
                      annotation_text=f"Barrier={barrier_level}")

    fig.add_hline(y=0, line_color="black", line_width=0.5)

    fig.update_layout(
        xaxis_title="Spot Price",
        yaxis_title="Payoff / P&L",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=400,
        margin=dict(l=0, r=0, t=0, b=0)
    )

    st.plotly_chart(fig, use_container_width=True)

# Sensitivity Analysis (only if results exist and checkbox is checked)
st.divider()

show_sensitivity = st.checkbox("Show Sensitivity Analysis", value=False)

if show_sensitivity and 'config_tuple' in st.session_state:
    st.subheader("Sensitivity Analysis")

    # Fewer points for faster calculation
    n_points = 10
    spot_range = np.linspace(underlying_price * 0.7, underlying_price * 1.3, n_points)

    tab1, tab2 = st.tabs(["Price vs Spot", "Delta vs Spot"])

    with tab1:
        with st.spinner("Calculating price sensitivity..."):
            sensitivity_results = calculate_sensitivity(
                st.session_state['config_tuple'],
                "underlying_price",
                spot_range.tolist(),
                compute_greeks=False
            )

        prices = [r['price'] for r in sensitivity_results]

        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(
            x=spot_range,
            y=prices,
            mode='lines+markers',
            name='Option Price',
            line=dict(color='#2ca02c', width=2)
        ))

        fig_price.add_vline(x=underlying_price, line_dash="dot", line_color="orange",
                           annotation_text=f"Current S={underlying_price}")
        fig_price.add_vline(x=strike_price, line_dash="dot", line_color="gray",
                           annotation_text=f"K={strike_price}")

        fig_price.update_layout(
            xaxis_title="Spot Price",
            yaxis_title="Option Price",
            height=350
        )

        st.plotly_chart(fig_price, use_container_width=True)

    with tab2:
        with st.spinner("Calculating Delta sensitivity..."):
            sensitivity_results = calculate_sensitivity(
                st.session_state['config_tuple'],
                "underlying_price",
                spot_range.tolist(),
                compute_greeks=True
            )

        deltas = [r['greeks']['delta'] if r['greeks'] else np.nan for r in sensitivity_results]

        fig_delta = go.Figure()
        fig_delta.add_trace(go.Scatter(
            x=spot_range,
            y=deltas,
            mode='lines+markers',
            name='Delta',
            line=dict(color='#1f77b4', width=2)
        ))

        fig_delta.add_vline(x=underlying_price, line_dash="dot", line_color="orange",
                           annotation_text=f"Current S={underlying_price}")
        fig_delta.add_vline(x=strike_price, line_dash="dot", line_color="gray",
                           annotation_text=f"K={strike_price}")

        fig_delta.update_layout(
            xaxis_title="Spot Price",
            yaxis_title="Delta",
            height=350
        )

        st.plotly_chart(fig_delta, use_container_width=True)

elif show_sensitivity:
    st.info("Calculate an option first to see sensitivity analysis.")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
    Option Calculator | Built with Streamlit<br>
    Pricing: Black-Scholes (European), LSM Monte Carlo (American, Asian, Barrier)
</div>
""", unsafe_allow_html=True)