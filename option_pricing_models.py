import numpy as np
import pandas as pd
import scipy.stats as st
import yfinance as yf
import streamlit as stl
import streamlit.components.v1 as components
import plotly.colors
import plotly.graph_objects as go
from plotly.subplots import make_subplots


############################################################
# Monte Carlo Pricing Model for European and Asian Options #
############################################################
def monte_carlo_option_pricing(S0, K, T, r, sigma, n, option_type):
    """
    Monte Carlo simulation to price European and Asian call options

    Parameters:
    - S0: Initial stock price
    - K: Strike price
    - T: Time to expiration (in years)
    - r: Risk-free interest rate
    - sigma: Volatility of the stock's returns
    - n: Number of simulations to run
    - option_type: Type of option ('European' or 'Asian')

    Returns:
    - option_price: The estimated price of the option
    """
    # Time step
    dt = T / n
    # Accumulator for the sum of payoffs
    payoff_sum = 0

    # Run the simulations
    for _ in range(n):
        # Initialise the stock price for each simulation
        S = S0
        # List to store the path of stock prices (used for Asian option)
        path = []

        # Simulate the path of the stock price over time
        for _ in range(int(T / dt)):
            # Update the stock price using the Geometric Brownian Motion formula
            S *= np.exp(
                (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.normal()
            )
            # Store the stock price in the path
            path.append(S)

        # Calculate the payoff for Asian option
        if option_type == "Asian":
            # Calculate the average stock price over the path
            average_price = np.mean(path)
            # Payoff for Asian call option
            payoff = max(average_price - K, 0)
        else:
            # Payoff for European call option
            payoff = max(S - K, 0)

        # Add the payoff to the cumulative sum
        payoff_sum += payoff

    # Discount the average payoff back to present value
    option_price = np.exp(-r * T) * (payoff_sum / n)
    return option_price


############################################################
# Binomial Pricing Model for European and American Options #
############################################################
def binomial_option_pricing(S0, K, T, r, sigma, n, option_type):
    """
    Binomial Tree model to price European and American call options

    Parameters:
    - S0: Initial stock price
    - K: Strike price
    - T: Time to expiration (in years)
    - r: Risk-free interest rate
    - sigma: Volatility of the stock's returns
    - n: Number of time steps in the binomial model
    - option_type: Type of option ('European' or 'American')

    Returns:
    - option_price: The estimated price of the option
    """

    # Time step
    dt = T / n
    # Up factor
    u = np.exp(sigma * np.sqrt(dt))
    # Down factor
    d = np.exp(-sigma * np.sqrt(dt))
    # Risk-neutral probability
    p = (np.exp(r * dt) - d) / (u - d)

    # Initialise asset prices at maturity
    prices = np.zeros(n + 1)
    prices[0] = S0 * (d**n)
    for i in range(1, n + 1):
        prices[i] = prices[i - 1] * (u / d)

    # Initialise option values at maturity
    values = np.maximum(0, prices - K)

    # Step backwards through the tree
    for j in range(n - 1, -1, -1):
        for i in range(j + 1):
            if option_type == "American":
                # Value for American option
                values[i] = np.maximum(
                    prices[i] - K,
                    np.exp(-r * dt) * (p * values[i + 1] + (1 - p) * values[i]),
                )
            else:
                # Value for European option
                values[i] = np.exp(-r * dt) * (p * values[i + 1] + (1 - p) * values[i])

    return values[0]


###############################
# Black-Scholes Pricing Model #
###############################
def black_scholes_option_pricing(S, K, T, r, sigma):
    """
    Black-Scholes model to price European call options

    Parameters:
    - S: Current stock price
    - K: Strike price
    - T: Time to expiration (in years)
    - r: Risk-free interest rate
    - sigma: Volatility of the stock's returns

    Returns:
    - call_price: The estimated price of the call option
    """

    # Calculate d1 and d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Calculate the call option price using the Black-Scholes formula
    call_price = S * st.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * st.norm.cdf(
        d2, 0.0, 1.0
    )
    return call_price


##############################################
# Greeks Calculation for Black-Scholes Model #
##############################################
def black_scholes_greeks(S, K, T, r, sigma):
    """
    Calculate the Greeks for a European call option using the Black-Scholes model

    Parameters:
    - S: Current stock price
    - K: Strike price
    - T: Time to expiration (in years)
    - r: Risk-free interest rate
    - sigma: Volatility of the stock's returns

    Returns:
    - greeks: A dictionary containing the values of Delta, Gamma, Theta, Vega, and Rho
    """

    # Calculate d1 and d2 using the Black-Scholes formula
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Calculate Delta
    delta = st.norm.cdf(d1)

    # Calculate Gamma
    gamma = st.norm.pdf(d1) / (S * sigma * np.sqrt(T))

    # Calculate Theta
    theta = -(S * st.norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(
        -r * T
    ) * st.norm.cdf(d2)

    # Calculate Vega
    vega = S * st.norm.pdf(d1) * np.sqrt(T)

    # Calculate Rho
    rho = K * T * np.exp(-r * T) * st.norm.cdf(d2)

    # Return the Greeks as a dictionary
    return {"Delta": delta, "Gamma": gamma, "Theta": theta, "Vega": vega, "Rho": rho}


################################################
#  Real Market Data Integration using yFinance #
################################################
def get_real_market_data(ticker):
    """
    Fetches real market data for a given ticker symbol using the yFinance library

    Parameters:
    - ticker: The ticker symbol of the stock

    Returns:
    - A dictionary with the current price and volatility of the stock if data is available
    - None if the ticker symbol is not valid or data is not available
    """

    # Initialise the Ticker object from yFinance
    stock = yf.Ticker(ticker)

    # Fetch the historical market data for the past year
    hist = stock.history(period="1y")

    # Check if the historical data is empty
    if hist.empty:
        stl.error("The ticker symbol is not valid or data is not available.")
        return None

    # Fetch the current price of the stock
    current_price = stock.history(period="1d")["Close"]

    # Check if the current price DataFrame is empty
    if current_price.empty:
        stl.error("Unable to fetch the current price for the ticker.")
        return None

    # Extract the latest closing price
    current_price = current_price.iloc[0]

    # Calculate the annualized volatility
    volatility = hist["Close"].pct_change().std() * np.sqrt(252)

    # Return the current price and volatility as a dictionary
    return {"current_price": current_price, "volatility": volatility}


##################################
# Parameter Sensitivity Analysis #
##################################
def parameter_sensitivity_analysis(model, S0, K, T, r, sigma, param, values):
    """
    Conducts a sensitivity analysis for an option pricing model by varying a specified parameter

    Parameters:
    - model: The option pricing model function to be analysed
    - S0: Initial stock price
    - K: Strike price
    - T: Time to expiration (in years)
    - r: Risk-free interest rate
    - sigma: Volatility of the underlying stock
    - param: The parameter to vary for the sensitivity analysis ('S0', 'K', 'T', 'r', 'sigma')
    - values: A list or array of values for the parameter being varied

    Returns:
    - A dictionary where keys are the varied parameter values and values are the corresponding option prices
    """
    
    # Initialise an empty dictionary to store results
    results = {}
    
    # Loop through each value in the specified range for the parameter
    for value in values:
        # Depending on which parameter is being varied, call the model with the adjusted parameter
        if param == "S0":
            results[value] = model(value, K, T, r, sigma)
        elif param == "K":
            results[value] = model(S0, value, T, r, sigma)
        elif param == "T":
            results[value] = model(S0, K, value, r, sigma)
        elif param == "r":
            results[value] = model(S0, K, T, value, sigma)
        elif param == "sigma":
            results[value] = model(S0, K, T, r, value)
    
    # Return the dictionary containing parameter values and corresponding option prices
    return results


#####################
# Scenario Analysis #
#####################
# def scenario_analysis(S0, K, T, r, sigma, scenarios):
def scenario_analysis(scenarios):
    results = {}
    for scenario, params in scenarios.items():
        price = black_scholes_option_pricing(
            params["S0"], params["K"], params["T"], params["r"], params["sigma"]
        )
        results[scenario] = price
    return results


#######################
# Streamlit Interface #
#######################
stl.title("Option Pricing Models")
stl.sidebar.header("User Input Parameters")

# Add horizontal menu at the top
menu = stl.selectbox("Section", ["Option Prices", "Scenario Analysis"], index=0)

# User inputs with labels including parameters
S0 = stl.sidebar.number_input("Initial stock price (S0)", value=100.0)
K = stl.sidebar.number_input("Strike price (K)", value=100.0)
T = stl.sidebar.number_input("Time to expiration in years (T)", value=1.0)
r = stl.sidebar.number_input("Risk-free interest rate (r)", value=0.05)
sigma = stl.sidebar.number_input("Volatility (σ)", value=0.2)
num_simulations = stl.sidebar.number_input(
    "Number of simulations (for Monte Carlo)", value=100, step=10
)
n = stl.sidebar.number_input("Number of time steps (for Binomial)", value=50)
option_type = stl.sidebar.selectbox("Option Type", ["European", "American", "Asian"])
ticker = stl.sidebar.text_input("Enter ticker for real market data")

# Fetch real market data if ticker is provided
if ticker:
    market_data = get_real_market_data(ticker)
    if market_data:
        S0 = market_data["current_price"]
        sigma = market_data["volatility"]
        stl.sidebar.write(f"Current Price: {S0:.2f}")
        stl.sidebar.write(f"Volatility: {sigma:.2f}")


# Greeks Surface Plots
def greeks_surface_plot(K_values, T_values, S0, r, sigma):
    greeks_list = ["Delta", "Gamma", "Theta", "Vega", "Rho"]
    fig = make_subplots(
        rows=2,
        cols=3,
        specs=[
            [{"type": "surface"}, {"type": "surface"}, {"type": "surface"}],
            [{"type": "surface"}, {"type": "surface"}, {"type": "surface"}],
        ],
        subplot_titles=greeks_list,
    )

    for i, greek in enumerate(greeks_list):
        greek_values = []
        for K in K_values:
            row = []
            for T in T_values:
                greek_value = black_scholes_greeks(S0, K, T, r, sigma)[greek]
                row.append(greek_value)
            greek_values.append(row)
        greek_values = np.array(greek_values).T

        fig.add_trace(
            go.Surface(
                z=greek_values,
                x=K_values,
                y=T_values,
                colorscale=plotly.colors.sequential.Plasma,
                colorbar=dict(
                    title=greek,
                    titlefont=dict(size=18),
                    tickfont=dict(size=14),
                    title_side="right",
                    outlinewidth=0,
                ),
                hovertemplate=f"T: %{{y}}<br>K: %{{x}}<br>{greek}: %{{z}}<extra></extra>",
            ),
            row=(i // 3) + 1,
            col=(i % 3) + 1,
        )

    fig.update_layout(height=800, title_text="Greeks Surface Plots")
    return fig


# Calculate option prices
with stl.spinner("Calculating option prices..."):
    colors = plotly.colors.qualitative.Plotly
    if option_type == "European":
        mc_price = monte_carlo_option_pricing(
            S0, K, T, r, sigma, num_simulations, "European"
        )
        binomial_price = binomial_option_pricing(S0, K, T, r, sigma, n, "European")
        bs_price = black_scholes_option_pricing(S0, K, T, r, sigma)
        greeks = black_scholes_greeks(S0, K, T, r, sigma)

        # Display results in a table
        stl.subheader("Option Prices")
        results = pd.DataFrame(
            {
                "Model": ["Monte Carlo", "Binomial", "Black-Scholes"],
                "Option Price": [mc_price, binomial_price, bs_price],
            }
        )

        # Convert dataframe to HTML
        html_table = results.to_html(classes="table", border=0, index=False)

        # Use components.html to display the HTML table
        components.html(
            f"""
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
        """,
            height=200,
        )

        # Display Greeks
        stl.subheader("Greeks for European Option")
        greeks_df = pd.DataFrame([greeks])

        # Convert dataframe to HTML
        html_table = greeks_df.to_html(classes="table", border=0, index=False)

        # Use components.html to display the HTML table
        components.html(
            f"""
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
        """,
            height=100,
        )

        # Visualising Greeks
        greek_names = list(greeks.keys())
        greek_values = list(greeks.values())

        fig = go.Figure(
            data=[
                go.Bar(
                    name="Greek Values",
                    x=greek_names,
                    y=greek_values,
                    marker_color=colors,
                    text=[f"{value:.3g}" for value in greek_values],
                    textposition="outside",
                    textfont=dict(size=18),
                )
            ]
        )
        fig.update_layout(
            margin=dict(l=10, r=10, t=0, b=10),
            xaxis=dict(tickfont=dict(size=16)),
            yaxis=dict(
                tickfont=dict(size=16),
            ),
            barcornerradius=30,
        )
        fig.update_traces(hoverinfo="none")
        stl.plotly_chart(fig)

        K_values = np.linspace(0.8 * K, 1.2 * K, 20)
        T_values = np.linspace(0.01, 2, 20)
        fig = greeks_surface_plot(K_values, T_values, S0, r, sigma)
        stl.plotly_chart(fig)

    elif option_type == "American":
        binomial_price = binomial_option_pricing(S0, K, T, r, sigma, n, "American")
        # Display results in a table
        stl.subheader("Option Prices")
        results = pd.DataFrame(
            {"Model": ["Binomial"], "Option Price": [binomial_price]}
        )
        # Convert dataframe to HTML
        html_table = results.to_html(classes="table", border=0, index=False)

        # Use components.html to display the HTML table
        components.html(
            f"""
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
        """,
            height=100,
        )

    elif option_type == "Asian":
        mc_asian_price = monte_carlo_option_pricing(
            S0, K, T, r, sigma, num_simulations, "Asian"
        )
        # Display results in a table
        stl.subheader("Option Prices")
        results = pd.DataFrame(
            {"Model": ["Monte Carlo"], "Option Price": [mc_asian_price]}
        )
        # Convert dataframe to HTML
        html_table = results.to_html(classes="table", border=0, index=False)

        # Use components.html to display the HTML table
        components.html(
            f"""
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
        """,
            height=100,
        )

# Sensitivity Analysis
sensitivity_labels = {
    "S0": "Initial stock price (S0)",
    "K": "Strike price (K)",
    "T": "Time to expiration (T)",
    "r": "Risk-free interest rate (r)",
    "sigma": "Volatility (σ)",
}
stl.subheader("Sensitivity Analysis of Option Price")
sensitivity_param = stl.sidebar.selectbox(
    "Sensitivity Analysis Parameter", list(sensitivity_labels.values())
)
sensitivity_key = [k for k, v in sensitivity_labels.items() if v == sensitivity_param][
    0
]
sensitivity_values = stl.sidebar.slider(
    f"Select range for {sensitivity_param}", 0.8 * S0, 1.2 * S0, (0.8 * S0, 1.2 * S0)
)
sensitivity_values = np.linspace(sensitivity_values[0], sensitivity_values[1], 10)
sensitivity_results = parameter_sensitivity_analysis(
    black_scholes_option_pricing,
    S0,
    K,
    T,
    r,
    sigma,
    sensitivity_key,
    sensitivity_values
)

# Visualising Sensitivity Analysis
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=list(sensitivity_results.keys()),
        y=list(sensitivity_results.values()),
        mode="lines+markers",
        marker_color=colors,
    )
)
# Adding annotations to key points
for key, value in sensitivity_results.items():
    fig.add_annotation(
        x=key, y=value, text=f"{value:.2f}", showarrow=False, yshift=15, xshift=-5
    )
fig.update_layout(
    title=f"with respect to {sensitivity_param}",
    xaxis=dict(
        title=sensitivity_param, tickfont=dict(size=16), titlefont=dict(size=18)
    ),
    yaxis=dict(title="Option Price", tickfont=dict(size=16), titlefont=dict(size=18)),
    margin=dict(l=10, r=10, t=30, b=10),
)
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

    fig = go.Figure(
        data=[
            go.Surface(
                z=prices,
                x=K_values,
                y=T_values,
                colorscale=plotly.colors.sequential.Agsunset,
                colorbar=dict(
                    title="Option Price",
                    titlefont=dict(size=18),
                    tickfont=dict(size=14),
                    title_side="right",
                    outlinewidth=0,
                ),
                hovertemplate="T: %{y}<br>K: %{x}<br>Option Price: %{z}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title="Strike Price (K)",
                titlefont=dict(size=18),
                tickfont=dict(size=14),
            ),
            yaxis=dict(
                title="Time to Expiration (T)",
                titlefont=dict(size=18),
                tickfont=dict(size=14),
            ),
            zaxis=dict(
                title="Option Price", titlefont=dict(size=18), tickfont=dict(size=14)
            ),
        ),
        margin=dict(l=10, r=10, t=0),
        height=600,
    )
    stl.plotly_chart(fig)


# Create a Volatility Surface plot
K_values = np.linspace(0.8 * K, 1.2 * K, 20)
T_values = np.linspace(0.01, 2, 20)
stl.subheader("Volatility Surface")
volatility_surface_plot(K_values, T_values, S0, r, sigma)

#####################
# Scenario Analysis #
#####################
# Define scenarios for stress testing
scenarios = {
    "Bull Market": {"S0": S0 * 1.2, "r": r, "sigma": sigma},
    "Bear Market": {"S0": S0 * 0.8, "r": r, "sigma": sigma},
    "High Volatility": {"S0": S0, "r": r, "sigma": sigma * 1.5},
    "Low Volatility": {"S0": S0, "r": r, "sigma": sigma * 0.5},
    "High Interest Rate": {"S0": S0, "r": r * 1.5, "sigma": sigma},
    "Low Interest Rate": {"S0": S0, "r": r * 0.5, "sigma": sigma},
}

# Perform Scenario Analysis
# scenario_results = scenario_analysis(S0, K, T, r, sigma, scenarios)
scenario_results = scenario_analysis(scenarios)

# Display Scenario Analysis
stl.subheader("Scenario Analysis")

# Scenario Comparison Plot
scenario_names = list(scenario_results.keys())
scenario_values = list(scenario_results.values())
fig = go.Figure(
    data=[
        go.Bar(
            name="Scenario Analysis",
            x=scenario_names,
            y=scenario_values,
            marker_color=colors,
            text=[f"{result:.2f}" for result in scenario_values],
            textposition="outside",
            textfont=dict(size=18),
        )
    ]
)
fig.update_layout(
    margin=dict(l=10, r=10, t=0, b=10),
    xaxis=dict(title="Scenario", tickfont=dict(size=14)),
    yaxis=dict(title="Option Price", tickfont=dict(size=14)),
)
fig.update_traces(hoverinfo="none")
stl.plotly_chart(fig)
