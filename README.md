# Performance_analysis
The complete stock performance analysis using python

The following script calls stock and benchmark indice price from yahoo finance
to create a complete performance report.

Calling the class:

# Positional Arguments
    assetTicker = Ticker symbol for the stock
    benchmarkTicker = Ticker symbol for benchmark
    performancePeriod = The time period in days of the investment period
    riskFreeRate = Annual risk free rate eg. 0.05 for 5%
    threshold = Annual investor return threshod for Omega ratio calculation
                Value is in percent

# Call eg:
    performanceAnalysis('MSFT','^GSPC','1000d',0.0025,0)

    # Outputs
        matplotlib plot performance summary
    1. Cummulative return graph of stock and benchmark
    2. Daily return plot.
    3. Histogram of return distribution including kde plot.
    4. Underwater plot of drawdown.
    5. Performance metrics.

# Libraries and versions
    pandas - 1.0.1
    numpy - 1.18.1
    yfinance - 0.1.54
    matplotlib - 3.1.3
    statsmodels - 0.11.0
    scipy - 1.4.1
    seaborn - 0.10.0


# Sample output
![Matplotlib output](/Figure_7.png)
