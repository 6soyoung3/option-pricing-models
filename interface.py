import pandas as pd
import streamlit as stl
import plotly.graph_objects as go
import plotly.colors
import plotly.express as px
import numpy as np
import streamlit.components.v1 as components
from option_pricing_models import *

#######################
# Streamlit Interface #
#######################
stl.title("Option Pricing Models")
stl.sidebar.header("User Input Parameters")

# User inputs with labels including parameters
ticker = stl.sidebar.text_input("Enter ticker for real market data")
# Fetch real market data if ticker is provided
if ticker:
    market_data, error = get_real_market_data(ticker)
    if market_data is None:
        stl.sidebar.error(error)
    else:
        default_S0 = market_data["current_price"]
        default_sigma = market_data["volatility"]
        stl.sidebar.write(f"Current Price: {default_S0:.2f}")
        stl.sidebar.write(f"Volatility: {default_sigma:.2f}")

        # Displays historical prices and volatility using line plots
        stl.subheader(f"Historical Data for {ticker.upper()}")
        hist_prices = market_data["historical_prices"]
        hist_volatility = market_data["historical_volatility"]

        fig_hist_prices = px.line(hist_prices, title="Historical Prices")
        fig_hist_prices.update_layout(
            margin=dict(l=10, r=10, t=25, b=10),
            xaxis=dict(title="Date", tickfont=dict(size=14)),
            yaxis=dict(title="Price", tickfont=dict(size=14)),
            showlegend=False
        )
        fig_hist_prices.update_traces(hovertemplate='Date=%{x}<br>Price=%{y}', name="")
        stl.plotly_chart(fig_hist_prices)

        # Plot historical volatility
        fig_hist_volatility = px.line(hist_volatility, title="Historical Volatility")
        fig_hist_volatility.update_layout(
            margin=dict(l=10, r=10, t=25, b=10),
            xaxis=dict(title="Date", tickfont=dict(size=14)),
            yaxis=dict(title="Volatility", tickfont=dict(size=14)),
            showlegend=False
        )
        fig_hist_volatility.update_traces(hovertemplate='Date=%{x}<br>Volatility=%{y}', name="")
        stl.plotly_chart(fig_hist_volatility)
else:
    # Sets default values for the initial stock price and volatility if no ticker is provided
    default_S0 = 100.0
    default_sigma = 0.2

S0 = stl.sidebar.number_input("Initial stock price (S0)", value=default_S0, step=5.0)
K = stl.sidebar.number_input("Strike price (K)", value=100.0, step=5.0)
T = stl.sidebar.number_input("Time to expiration in years (T)", value=1.0, step=0.25)
r = stl.sidebar.number_input("Risk-free interest rate (r)", value=0.05, step=0.025)
sigma = stl.sidebar.number_input("Volatility (σ)", value=default_sigma, step=0.05)
num_simulations = stl.sidebar.number_input(
    "Number of simulations (for Monte Carlo)", value=100, step=10
)
n = stl.sidebar.number_input("Number of time steps (for Binomial)", value=50)


# Calculate option prices
with stl.spinner("Calculating option prices..."):
    colors = plotly.colors.qualitative.Plotly
    mc_price = monte_carlo_option_pricing(S0, K, T, r, sigma, num_simulations)
    binomial_price = binomial_option_pricing(S0, K, T, r, sigma, n)
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

    # Stock prices for the plot
    stock_prices = np.linspace(S0//5, S0*2, 20)

    # Calculate option prices using different models
    mc_prices = [monte_carlo_option_pricing(S, K, T, r, sigma, num_simulations) for S in stock_prices]
    binomial_prices = [binomial_option_pricing(S, K, T, r, sigma, n) for S in stock_prices]
    bs_prices = [black_scholes_option_pricing(S, K, T, r, sigma) for S in stock_prices]
    intrinsic_values = [max(S - K, 0) for S in stock_prices]

    # Create plot
    fig = go.Figure()

    # Add Monte Carlo line
    fig.add_trace(go.Scatter(x=stock_prices, y=mc_prices, mode='lines+markers', name='Monte Carlo', line_color=colors[1], hovertemplate='MC: %{y:.3f}<extra></extra>'))

    # Add Binomial line
    fig.add_trace(go.Scatter(x=stock_prices, y=binomial_prices, mode='lines+markers', name='Binomial', line_color=colors[2], hovertemplate='B: %{y:.3f}<extra></extra>'))

    # Add Black-Scholes line
    fig.add_trace(go.Scatter(x=stock_prices, y=bs_prices, mode='lines+markers', name='Black-Scholes', line_color=colors[3], hovertemplate='BS: %{y:.3f}<extra></extra>'))

    # Add Intrinsic Values lines
    fig.add_trace(go.Scatter(x=stock_prices, y=intrinsic_values, mode='lines+markers', name='Intrinsic Value', line=dict(color='grey', dash='dash'), hovertemplate='IV: %{y:.3f}<extra></extra>'))

    # Add annotations for Intrinsic Value and regions
    fig.add_annotation(x=K, y=0, text="E", showarrow=True, arrowhead=2, ax=0, ay=-40)
    fig.add_shape(type="rect", x0=S0//5, y0=0, x1=K, y1=max(bs_prices), line_width=0, fillcolor="LightGrey", opacity=0.3)
    fig.add_shape(type="rect", x0=K, y0=0, x1=S0*2, y1=max(bs_prices), line_width=0)
    fig.add_annotation(x=S0//5+(K-S0//5)//2, y=-5, text="Out", showarrow=False)
    fig.add_annotation(x=S0*2, y=-5, text="In", showarrow=False)
    fig.add_annotation(x=K, y=-5, text="At", showarrow=False)

    # Update layout
    fig.update_layout(
        margin=dict(l=10, r=10, t=0, b=10),
        xaxis=dict(title="Stock Price", tickfont=dict(size=14)),
        yaxis=dict(title="Call Option Value", tickfont=dict(size=14)),  
        hovermode='x unified',
        showlegend=True
    )

    # Display the plot in Streamlit
    stl.plotly_chart(fig)


# Display Greeks
with stl.spinner("Displaying Greeks..."):
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

    # Visualising the Greeks (Delta, Gamma, Theta, Vega, Rho) in a table and a bar chart
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

    # K_values = np.linspace(0.8 * K, 1.2 * K, 20)
    # T_values = np.linspace(0.01, 2, 20)
    # fig = greeks_surface_plot(K_values, T_values, S0, r, sigma)
    # stl.plotly_chart(fig)
