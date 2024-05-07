'''
to generate valid tickers for the stock market, namely filtering out the tickers that are not in the stock market
'''
import numpy as np
import pandas as pd
import json
# path_amex = "C:\\Users\\AC\\Desktop\\Quantum Computation\\TSC\\SAPO-main\\SAPO-main\\get_data_US\\amex_data.csv"
# stock_file = path_amex
# stock_data = pd.read_csv(
#     filepath_or_buffer=stock_file
# )
# stock_data = stock_data.drop([1])
# stock_data = stock_data.dropna(axis='columns')
# stock_data.to_csv("amex_data_tailored.csv")

tailored_amex_path = "amex_data_tailored.csv"
valid_amex_tickers_path = "valid_amex_tickers.json"
data = pd.read_csv(tailored_amex_path)
print(data.columns)
col = data.columns.tolist()
# 2nd column is the ticker, 2+6 is the second ticker
n = len(col)
valid_tickers = []
for i in range(2, n, 6):
    valid_tickers.append(col[i])

print(valid_tickers)
json.dump(valid_tickers, open(valid_amex_tickers_path, "w"))
