# Option Pricing Models

## Introduction

This project implements various option pricing models including Monte Carlo simulations, Binomial Tree models, and the Black-Scholes model. It also includes a web-based interface built using Streamlit to visualise and interact with these models. The project solves the problem of option pricing by providing multiple methods to estimate the price of European call options. It also offers a user-friendly interface to visualise these models and their results, making it accessible for both finance professionals and academics. The project aims to showcase practical knowledge in quantitative finance, particularly in the area of derivatives pricing.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Future Work](#future-work)

## Project Structure

The project consists of two main files:

1. `option_pricing_models.py`: Contains the implementation of different option pricing models and related utility functions.
2. `interface.py`: Provides a Streamlit-based user interface for interacting with the option pricing models.

### Option Pricing Models

>#### Monte Carlo Simulation

Monte Carlo simulation is a numerical method used to estimate the price of an option by simulating the random paths of the underlying asset's price.

**Mathematical Formula:**

The stock price at time $T$ is given by:

$$S_T = S_0 \exp\left(\left(r - \frac{\sigma^2}{2}\right)T + \sigma \sqrt{T} Z\right)$$

where $Z$ is a random variable from a standard normal distribution.

The option price is then estimated as the discounted average of the payoffs:

 $$\text{Option Price} = e^{-rT} \frac{1}{N} \sum_{i=1}^{N} \max(S_T^{(i)} - K, 0)$$

>#### Binomial Tree Model

The Binomial Tree model is a discrete-time model for pricing options by constructing a tree of possible future stock prices.

**Mathematical Formula:**

In each time step, the stock price can move up by a factor $u$ or down by a factor $d$:

$$u = e^{\sigma \sqrt{\Delta t}}$$

$$d = e^{-\sigma \sqrt{\Delta t}}$$

The risk-neutral probability $p$ is given by:

$$p = \frac{e^{r \Delta t} - d}{u - d}$$

The option price is computed by working backward from the final nodes to the present value:
$$V = e^{-r \Delta t} (p V_u + (1 - p) V_d)$$

>#### Black-Scholes Model

The Black-Scholes model provides a closed-form solution for the price of European call and put options.

**Mathematical Formula:**

The call option price is given by:

$$C = S_0 N(d_1) - K e^{-rT} N(d_2)$$

where

$$d_1 = \frac{\ln(S_0 / K) + (r + \sigma^2 / 2)T}{\sigma \sqrt{T}}$$

$$d_2 = d_1 - \sigma \sqrt{T}$$

$N(\cdot)$ is the cumulative distribution function of the standard normal distribution.

>#### Greeks

The Greeks measure the sensitivity of the option price to various factors:
- **Delta ($\delta$):** Sensitivity to the underlying asset's price.
- **Gamma ($\gamma$):** Sensitivity of Delta to the underlying asset's price.
- **Theta ($\theta$):** Sensitivity to the passage of time.
- **Vega ($\nu$):** Sensitivity to the underlying asset's volatility.
- **Rho ($\rho$):** Sensitivity to the risk-free interest rate.

### Streamlit Interface

The Streamlit interface allows users to input parameters, fetch real market data, and visualise the pricing models and Greeks.

#### Features:
- Input parameters such as initial stock price, strike price, time to expiration, risk-free rate, and volatility.
- Fetch real-time market data using the yFinance API.
- Display historical prices and volatility.
- Visualise option prices calculated using different models.
- Display the Greeks for the Black-Scholes model.

## Installation

### Prerequisites

Ensure you have 
- Python 3.x
- pip (Python package installer)

and the following Python packages installed:
- numpy
- scipy
- yfinance
- pandas
- streamlit
- plotly

### Running the Application

To run the Streamlit application, use the following command:

```bash
streamlit run interface.py
```

This will start the Streamlit server and open the application in your web browser.

## Usage

### Instructions
1. **Input Parameters**: Enter the initial stock price, strike price, time to expiration, risk-free rate, and volatility.
2. **Fetch Market Data**: Optionally, enter a ticker symbol to fetch real market data.
3. **Calculate Option Prices**: View the option prices computed using different models.
4. **Visualise Data**: Explore the historical prices, volatility, and Greeks using interactive plots.

Here's a simple example of using the Monte Carlo simulation to price an option:

```python
from option_pricing_models import monte_carlo_option_pricing

S0 = 100  # Initial stock price
K = 105   # Strike price
T = 1     # Time to expiration in years
r = 0.05  # Risk-free interest rate
sigma = 0.2  # Volatility
n = 10000  # Number of simulations

option_price = monte_carlo_option_pricing(S0, K, T, r, sigma, n)
print(f"The estimated option price is: {option_price:.2f}")
```

### Features
- **Monte Carlo Simulation**: Estimates option prices by simulating multiple paths of stock prices.
- **Binomial Tree Model**: Uses a discrete-time model to price options.
- **Black-Scholes Model**: Provides a closed-form solution for pricing European call options.
- **Greeks Calculation**: Computes Delta, Gamma, Theta, Vega, and Rho.
- **Real Market Data Integration**: Fetches live data using yFinance.
- **Interactive Plots**: Visualises historical prices, volatility, and sensitivity analysis.

## Future Work

Potential improvements and additions to this project could include:
- Adding scenario analysis and stress testing to simulate extreme market conditions and see how option prices and Greeks respond.
- Adding 3D volatility surface plot showing implied volatility across different strike prices and maturities.
- Adding more types of options (e.g., American options, Asian options).
- Enhancing the user interface with more interactive features and better visualisations (e.g., allowing users to hover over data points for additional information, zoom in/out on charts, and dynamically adjust parameters).
