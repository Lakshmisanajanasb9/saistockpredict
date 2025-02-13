import yfinance as yf 
import pandas as pd 
from newsapi import NewsApiClient


def get_stock(ticker):
    yfObject = yf.Ticker(ticker)
    data = yfObject.history(period="1y")
    return data 


# Example: Fetch data for AAPL
ticker = "AAPL"
start_date = "2020-01-01"
end_date = "2025-01-01"
stock_data = get_stock(ticker, start_date, end_date)

# Display the stock data
stock_data.head()


def fetch_news(ticker):
    newsapi = NewsApiClient(api_key='21cb2dc632e64ae797390a479b7ae2e1')
    top_headlines = newsapi.get_top_headlines(q='bitcoin',
                                          #sources='bbc-news,the-verge',
                                          category='business',
                                          language='en',
                                          country='us')


# Function to download stock data from Yahoo Finance and curate it
def download_stock_data(tickers, start_date, end_date):
    # Create an empty list to hold DataFrames
    stock_data_list = []

    # Loop through each ticker and download the data
    for ticker in tickers:
        print(f"Downloading data for {ticker}...")

        # Fetch stock data for the ticker between start and end dates
        data = yf.download(ticker, start=start_date, end=end_date)

        # Add the 'Ticker' column for identification
        data['Ticker'] = ticker

        # Append to the list
        stock_data_list.append(data)

    # Concatenate all DataFrames into one
    combined_data = pd.concat(stock_data_list)

    # Reset the index so the date is a column
    combined_data.reset_index(inplace=True)

    # Print a small sample of the data to check
    print(combined_data.head())

    # Save the data to a CSV file
    combined_data.to_csv('curated_stock_data.csv', index=False)
    print("Data saved to 'curated_stock_data.csv'")

# Define the stock tickers you want to fetch data for
tickers = ['AAPL', 'GOOG', 'AMZN', 'MSFT', 'TSLA']  # Add any tickers you need

# Define the start and end date for historical data
start_date = '2010-01-01'
end_date = '2025-01-01'

# Download and curate stock data
#download_stock_data(tickers, start_date, end_date)
