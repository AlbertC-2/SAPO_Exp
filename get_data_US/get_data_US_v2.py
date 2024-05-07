import json
import yfinance as yf
import os
import sys
os.getcwd()

# Define the ticker symbol for the stock market index you want to fetch data for
# List of ticker symbols
nasdaq_tickers_path = "C:\\Users\\AC\\Desktop\\Quantum Computation\\TSC\\SAPO-main\\SAPO-main\\get_data_US\\nasdaq_tickers.json"
amex_tickers_path = "C:\\Users\\AC\\Desktop\\Quantum Computation\\TSC\\SAPO-main\\SAPO-main\\get_data_US\\amex_tickers.json"
nyse_tickers_path = "C:\\Users\\AC\\Desktop\\Quantum Computation\\TSC\\SAPO-main\\SAPO-main\\get_data_US\\nyse_tickers.json"

nasdaq_data_path = "C:\\Users\\AC\\Desktop\\Quantum Computation\\TSC\\SAPO-main\\SAPO-main\\get_data_US\\nasdaq_data.csv"
amex_data_path = "C:\\Users\\AC\\Desktop\\Quantum Computation\\TSC\\SAPO-main\\SAPO-main\\get_data_US\\amex_data.csv"
nyse_data_path = "C:\\Users\\AC\\Desktop\\Quantum Computation\\TSC\\SAPO-main\\SAPO-main\\get_data_US\\nyse_data.csv"


def get_tickers(tickers_path):
    with open(tickers_path) as f:
        ticker_symbols = json.load(f)
    return ticker_symbols


def download_data(tickers_path, data_path):
    '''
    download data from yfinance, save it to a csv file
    tickers_path: path to the json file containing the ticker symbols
    data_path: path to the csv file where the data will be saved
    '''
    ticker_symbols = get_tickers(tickers_path)
    # Define the start and end dates for the data
    start_date = "2022-01-01"
    end_date = "2023-01-01"
    # Fetch the data using yfinance
    data = yf.download(ticker_symbols, start=start_date,
                       end=end_date, group_by="tickers")
    # store the data in a csv file
    data.to_csv(data_path)


if __name__ == "__main__":
    if sys.argv[1] == "nasdaq":
        download_data(nasdaq_tickers_path, nasdaq_data_path)
    elif sys.argv[1] == "amex":
        download_data(amex_tickers_path, amex_data_path)
    elif sys.argv[1] == "nyse":
        download_data(nyse_tickers_path, nyse_data_path)
