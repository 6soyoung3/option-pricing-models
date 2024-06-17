import numpy as np
import pandas as pd
import scipy.stats as st
import yfinance as yf
import streamlit as stl
import streamlit.components.v1 as components
import plotly.colors
import plotly.graph_objects as go

# Monte Carlo Pricing Model for European Options
def monte_carlo_option_pricing(S0, K, T, r, sigma, num_simulations):
    dt = T / num_simulations
    payoff_sum = 0

    for _ in range(num_simulations):
        S = S0
        for _ in range(int(T / dt)):
            S *= np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * np.random.normal())
        payoff_sum += max(S - K, 0)

    option_price = np.exp(-r * T) * (payoff_sum / num_simulations)
    return option_price

# Binomial Pricing Model for European and American Options
def binomial_option_pricing(S0, K, T, r, sigma, n, option_type="European"):
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = np.exp(-sigma * np.sqrt(dt))
    p = (np.exp(r * dt) - d) / (u - d)

    prices = np.zeros(n + 1)
    prices[0] = S0 * (d ** n)
    for i in range(1, n + 1):
        prices[i] = prices[i - 1] * (u / d)

    values = np.maximum(0, prices - K)

    for j in range(n - 1, -1, -1):
        for i in range(j + 1):
            if option_type == "American":
                values[i] = np.maximum(prices[i] - K, np.exp(-r * dt) * (p * values[i + 1] + (1 - p) * values[i]))
            else:
                values[i] = np.exp(-r * dt) * (p * values[i + 1] + (1 - p) * values[i])

    return values[0]

# Black-Scholes Pricing Model
def black_scholes_option_pricing(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_price = (S * st.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * st.norm.cdf(d2, 0.0, 1.0))
    return call_price

# Greeks Calculation for Black-Scholes Model
def black_scholes_greeks(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    delta = st.norm.cdf(d1)
    gamma = st.norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = -(S * st.norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * st.norm.cdf(d2)
    vega = S * st.norm.pdf(d1) * np.sqrt(T)
    rho = K * T * np.exp(-r * T) * st.norm.cdf(d2)

    return {
        'Delta': delta,
        'Gamma': gamma,
        'Theta': theta,
        'Vega': vega,
        'Rho': rho
    }

# Real Market Data Integration using yFinance
def get_real_market_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")
    current_price = stock.history(period="1d")['Close'].iloc[0]
    volatility = hist['Close'].pct_change().std() * np.sqrt(252)
    return {
        'current_price': current_price,
        'volatility': volatility
    }

# Parameter Sensitivity Analysis
def parameter_sensitivity_analysis(model, S0, K, T, r, sigma, param, values, **kwargs):
    results = {}
    for value in values:
        if param == 'S0':
            results[value] = model(value, K, T, r, sigma, **kwargs)
        elif param == 'K':
            results[value] = model(S0, value, T, r, sigma, **kwargs)
        elif param == 'T':
            results[value] = model(S0, K, value, r, sigma, **kwargs)
        elif param == 'r':
            results[value] = model(S0, K, T, value, sigma, **kwargs)
        elif param == 'sigma':
            results[value] = model(S0, K, T, r, value, **kwargs)
    return results

# Monte Carlo Pricing Model for Asian Options
def monte_carlo_asian_option_pricing(S0, K, T, r, sigma, num_simulations):
    dt = T / num_simulations
    payoff_sum = 0

    for _ in range(num_simulations):
        S = S0
        path = []
        for _ in range(int(T / dt)):
            S *= np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * np.random.normal())
            path.append(S)
        average_price = np.mean(path)
        payoff_sum += max(average_price - K, 0)

    option_price = np.exp(-r * T) * (payoff_sum / num_simulations)
    return option_price

# Streamlit Interface
stl.title("Option Pricing Models")
stl.sidebar.header("User Input Parameters")

# User inputs with labels including parameters
S0 = stl.sidebar.number_input("Initial stock price (S0)", value=100.0)
K = stl.sidebar.number_input("Strike price (K)", value=100.0)
T = stl.sidebar.number_input("Time to expiration in years (T)", value=1.0)
r = stl.sidebar.number_input("Risk-free interest rate (r)", value=0.05)
sigma = stl.sidebar.number_input("Volatility (σ)", value=0.2)
num_simulations = stl.sidebar.number_input("Number of simulations (for Monte Carlo)", value=100, step=10)
n = stl.sidebar.number_input("Number of time steps (for Binomial)", value=50)
option_type = stl.sidebar.selectbox("Option Type", ["European", "American", "Asian"])
ticker = stl.sidebar.text_input("Enter ticker for real market data")

# Fetch real market data if ticker is provided
if ticker:
    market_data = get_real_market_data(ticker)
    S0 = market_data['current_price']
    sigma = market_data['volatility']
    stl.sidebar.write(f"Current Price: {S0:.2f}")
    stl.sidebar.write(f"Volatility: {sigma:.2f}")

# Calculate option prices
with stl.spinner("Calculating option prices..."):
    colors = plotly.colors.qualitative.Plotly
    if option_type == "European":
        mc_price = monte_carlo_option_pricing(S0, K, T, r, sigma, num_simulations)
        binomial_price = binomial_option_pricing(S0, K, T, r, sigma, n, option_type="European")
        bs_price = black_scholes_option_pricing(S0, K, T, r, sigma)
        greeks = black_scholes_greeks(S0, K, T, r, sigma)

        # Display results in a table
        stl.subheader("Option Prices")
        results = pd.DataFrame({
            "Model": ["Monte Carlo", "Binomial", "Black-Scholes"],
            "Option Price": [mc_price, binomial_price, bs_price]
        })

        # Convert dataframe to HTML
        html_table = results.to_html(classes='table', border=0, index=False)

        # Use components.html to display the HTML table
        components.html(f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&display=swap');
            .table {{
                width: 100%;
                color: #212529;
                font-family: 'IBM Plex Sans', sans-serif;
                border-collapse: collapse;
            }}
            .table th,
            .table td {{
                padding: 0.75rem;
                border-top: 1px solid #dee2e6;
                text-align: center;
            }}
        </style>
        {html_table}
        """, height=200)

        # Display Greeks
        stl.subheader("Greeks for European Option")
        greeks_df = pd.DataFrame([greeks])

        # Convert dataframe to HTML
        html_table = greeks_df.to_html(classes='table', border=0, index=False)

        # Use components.html to display the HTML table
        components.html(f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&display=swap');
            .table {{
                width: 100%;
                color: #212529;
                font-family: 'IBM Plex Sans', sans-serif;   
                border-collapse: collapse;     
            }}
            .table th,
            .table td {{
                border-top: 1px solid #dee2e6;
                padding: 0.75rem;
                text-align: center;
            }}
        </style>
        {html_table}
        """, height=100)

        # Visualising Greeks
        greek_names = list(greeks.keys())
        greek_values = list(greeks.values())

        fig = go.Figure(data=[
            go.Bar(
                name='Greek Values',
                x=greek_names,
                y=greek_values,
                marker_color=colors,
                text=[f'{value:.3g}' for value in greek_values],
                textposition='outside', 
                textfont=dict(size=18)
            )
        ])
        fig.update_layout(
            margin=dict(l=10, r=10, t=0, b=10),
            xaxis=dict(
                tickfont=dict(size=16)
            ),
            yaxis=dict(
                tickfont=dict(size=16),
            ),
            barcornerradius=30)
        fig.update_traces(hoverinfo='none')
        stl.plotly_chart(fig)

    elif option_type == "American":
        binomial_price = binomial_option_pricing(S0, K, T, r, sigma, n, option_type="American")
        # Display results in a table
        stl.subheader("Option Prices")
        results = pd.DataFrame({
            "Model": ["Binomial"],
            "Option Price": [binomial_price]
        })
        # Convert dataframe to HTML
        html_table = results.to_html(classes='table', border=0, index=False)

        # Use components.html to display the HTML table
        components.html(f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&display=swap');
            .table {{
                width: 100%;
                color: #212529;
                font-family: 'IBM Plex Sans', sans-serif;
                border-collapse: collapse;
            }}
            .table th,
            .table td {{
                padding: 0.75rem;
                border-top: 1px solid #dee2e6;
                text-align: center;
            }}
        </style>
        {html_table}
        """, height=100)
        # stl.table(results)

    elif option_type == "Asian":
        mc_asian_price = monte_carlo_asian_option_pricing(S0, K, T, r, sigma, num_simulations)
        # Display results in a table
        stl.subheader("Option Prices")
        results = pd.DataFrame({
            "Model": ["Monte Carlo"],
            "Option Price": [mc_asian_price]
        })
        # Convert dataframe to HTML
        html_table = results.to_html(classes='table', border=0, index=False)

        # Use components.html to display the HTML table
        components.html(f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&display=swap');
            .table {{
                width: 100%;
                color: #212529;
                font-family: 'IBM Plex Sans', sans-serif;
                border-collapse: collapse;
            }}
            .table th,
            .table td {{
                padding: 0.75rem;
                border-top: 1px solid #dee2e6;
                text-align: center;
            }}
        </style>
        {html_table}
        """, height=100)
        # stl.table(results)

# Sensitivity Analysis
sensitivity_labels = {
    'S0': 'Initial stock price (S0)',
    'K': 'Strike price (K)',
    'T': 'Time to expiration (T)',
    'r': 'Risk-free interest rate (r)',
    'sigma': 'Volatility (σ)'
}
stl.subheader("Sensitivity Analysis of Option Price")
sensitivity_param = stl.sidebar.selectbox("Sensitivity Analysis Parameter", list(sensitivity_labels.values()))
sensitivity_key = [k for k, v in sensitivity_labels.items() if v == sensitivity_param][0]
sensitivity_values = stl.sidebar.slider(f"Select range for {sensitivity_param}", 0.8 * S0, 1.2 * S0, (0.8 * S0, 1.2 * S0))
sensitivity_values = np.linspace(sensitivity_values[0], sensitivity_values[1], 10)
sensitivity_results = parameter_sensitivity_analysis(black_scholes_option_pricing, S0, K, T, r, sigma, sensitivity_key, sensitivity_values)

# Visualising Sensitivity Analysis
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=list(sensitivity_results.keys()),
    y=list(sensitivity_results.values()),
    mode='lines+markers',
    marker_color=colors
))
fig.update_layout(
    title=f'with respect to {sensitivity_param}',
    xaxis=dict(
                title = sensitivity_param,
                tickfont=dict(size=16),
                titlefont=dict(size=18)
            ),
            yaxis=dict(
                title = 'Option Price',
                tickfont=dict(size=16),
                titlefont=dict(size=18)
            ),
    margin=dict(l=10, r=10, t=30, b=10))
stl.plotly_chart(fig)

# Volatility Surface Plot
def volatility_surface_plot(K_values, T_values, S0, r, sigma):
    prices = []
    for K in K_values:
        row = []
        for T in T_values:
            price = black_scholes_option_pricing(S0, K, T, r, sigma)
            row.append(price)
        prices.append(row)
    K_grid, T_grid = np.meshgrid(K_values, T_values)
    prices = np.array(prices).T

    fig = go.Figure(data=[go.Surface(
        z=prices,
        x=K_values,
        y=T_values,
        colorscale = plotly.colors.sequential.Agsunset,
        colorbar=dict(
            title='Option Price',
            titlefont=dict(size=18),
            tickfont=dict(size=14),
            title_side="right",
            outlinewidth=0),
            hovertemplate='T: %{y}<br>K: %{x}<br>Option Price: %{z}<extra></extra>'
            )
        ]
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title='Strike Price (K)',
                titlefont=dict(size=18),
                tickfont=dict(size=14)
            ),
            yaxis=dict(
                title='Time to Expiration (T)',
                titlefont=dict(size=18),
                tickfont=dict(size=14)
            ),
            zaxis=dict(
                title='Option Price',
                titlefont=dict(size=18),
                tickfont=dict(size=14)
            ),
        ),
        margin=dict(l=10, r=10, t=0),
        height=600
    )
    stl.plotly_chart(fig)

# Create a Volatility Surface plot
K_values = np.linspace(0.8 * K, 1.2 * K, 20)
T_values = np.linspace(0.01, 2, 20)
stl.subheader("Volatility Surface")
volatility_surface_plot(K_values, T_values, S0, r, sigma)
