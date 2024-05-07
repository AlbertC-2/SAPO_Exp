import json
import yfinance as yf
import os
os.getcwd()

# Define the ticker symbol for the stock market index you want to fetch data for
# List of ticker symbols
# replace with your list
with open("C:\\Users\\AC\\Desktop\\Quantum Computation\\TSC\\SAPO-main\\SAPO-main\\get_data_US\\nasdaq_tickers.json") as f:
    ticker_symbols = json.load(f)
# ticker_symbols = ["AAPL", "MSFT", "GOOGL"]
# Define the start and end dates for the data
start_date = "2022-01-01"
end_date = "2023-01-01"
# Fetch the data using yfinance
data = yf.download(ticker_symbols, start=start_date, end=end_date)
# store the data in a csv file
data.to_csv(
    "C:\\Users\\AC\\Desktop\\Quantum Computation\\TSC\\SAPO-main\\SAPO-main\\get_data_US\\nasdaq_data.csv")


# for ticker_symbol in ticker_symbols:
#     data = yf.download(ticker_symbol, start=start_date, end=end_date)

#     # Print the fetched data
#     print(f"\nData for {ticker_symbol}:")
#     print(data)
#     # store the data in a csv file
#     data.to_csv(f"{ticker_symbol}.csv")
# Print the fetched data
# print(data)
# import yfinance as yf

# Dictionary to store the data for each ticker
data_dict = {}
# ticker_symbols = ["AAGR"]
# print(ticker_symbols)

# for symbol in ticker_symbols:
#     try:
#         data = yf.download(symbol, start=start_date, end=end_date)
#         data_dict[symbol] = data
#     except Exception as e:
#         print(f"Failed to fetch data for {symbol}: {e}")
# print(data_dict["AIRT"])

# # Define the start and end dates for the data
# start_date = "2017-01-01"
# end_date = "2022-01-01"

# # List of ticker symbols
# ticker_symbols = ["AAPL", "MSFT", "GOOGL", "..."]  # replace with your list

# # Fetch the data using yfinance
# for ticker_symbol in ticker_symbols:
#     data = yf.download(ticker_symbol, start=start_date, end=end_date)

#     # Print the fetched data
#     print(f"\nData for {ticker_symbol}:")
#     print(data)
