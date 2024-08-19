import numpy as np
import scipy.stats as st
import yfinance as yf


def monte_carlo_option_pricing(S0, K, T, r, sigma, n):
    """
    Monte Carlo simulation to price European call options

    Parameters:
    - S0: Initial stock price
    - K: Strike price
    - T: Time to expiration (in years)
    - r: Risk-free interest rate
    - sigma: Volatility of the stock's returns
    - n: Number of simulations to run

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

        # Simulate the path of the stock price over time
        for _ in range(int(T / dt)):
            # Update the stock price using the Geometric Brownian Motion formula
            S *= np.exp(
                (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.normal()
            )

        # Payoff for European call option
        payoff = max(S - K, 0)

        # Add the payoff to the cumulative sum
        payoff_sum += payoff

    # Discount the average payoff back to present value
    option_price = np.exp(-r * T) * (payoff_sum / n)
    return option_price


def binomial_option_pricing(S0, K, T, r, sigma, n):
    """
    Binomial Tree model to price European call options

    Parameters:
    - S0: Initial stock price
    - K: Strike price
    - T: Time to expiration (in years)
    - r: Risk-free interest rate
    - sigma: Volatility of the stock's returns
    - n: Number of time steps in the binomial model

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
            values[i] = np.exp(-r * dt) * (p * values[i + 1] + (1 - p) * values[i])

    return values[0]


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
        # stl.error("The ticker symbol is not valid or data is not available.")
        return None, "The ticker symbol is not valid or data is not available."

    # Fetch the current price of the stock
    current_price = stock.history(period="1d")["Close"]

    # Check if the current price DataFrame is empty
    if current_price.empty:
        # stl.error("Unable to fetch the current price for the ticker.")
        return None, "Unable to fetch the current price for the ticker."

    # Extract the latest closing price
    current_price = current_price.iloc[0]

    # Calculate the annualised volatility
    volatility = hist["Close"].pct_change().std() * np.sqrt(252)

    # Calculate historical volatility (rolling 30-day)
    hist_volatility = hist["Close"].pct_change().rolling(window=30).std() * np.sqrt(252)

     # Return the current price, volatility, historical prices, and historical volatility as a dictionary
    return {
        "current_price": current_price,
        "volatility": volatility,
        "historical_prices": hist["Close"],
        "historical_volatility": hist_volatility
    }, None


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
