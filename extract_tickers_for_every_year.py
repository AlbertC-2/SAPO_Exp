"""
helper script for nasdaq100.py
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Callable, Union
import math

"""
https://en.wikipedia.org/wiki/Nasdaq-100
https://stockmarketmba.com/stocksinthenasdaq100.php
"""
def _get_nasdaq100() -> Tuple[str]:
    return (
        "AAPL",
        "ABNB",
        "ADBE",
        "ADI",
        "ADP",
        "ADSK",
        "AEP",
        "ALGN",
        "AMAT",
        "AMD",
        "AMGN",
        "AMZN",
        "ANSS",
        "ASML",
        "ATVI",
        "AVGO",
        "AZN",
        "BIIB",
        "BKNG",
        "BKR",
        "CDNS",
        "CEG",
        "CHTR",
        "CMCSA",
        "COST",
        "CPRT",
        "CRWD",
        "CSCO",
        "CSGP",
        "CSX",
        "CTAS",
        "CTSH",
        "DDOG",
        "DLTR",
        "DXCM",
        "EA",
        "EBAY",
        "ENPH",
        "EXC",
        "FANG",
        "FAST",
        "FISV",
        "FTNT",
        "GILD",
        "GOOG",
        "GOOGL",
        "HON",
        "IDXX",
        "ILMN",
        "INTC",
        "INTU",
        "ISRG",
        "JD",
        "KDP",
        "KHC",
        "KLAC",
        "LCID",
        "LRCX",
        "LULU",
        "MAR",
        "MCHP",
        "MDLZ",
        "MELI",
        "META",
        "MNST",
        "MRNA",
        "MRVL",
        "MSFT",
        "MU",
        "NFLX",
        "NVDA",
        "NXPI",
        "ODFL",
        "ORLY",
        "PANW",
        "PAYX",
        "PCAR",
        "PDD",
        "PEP",
        "PYPL",
        "QCOM",
        "REGN",
        "RIVN",
        "ROST",
        "SBUX",
        "SGEN",
        "SIRI",
        "SNPS",
        "TEAM",
        "TMUS",
        "TSLA",
        "TXN",
        "VRSK",
        "VRTX",
        "WBA",
        "WBD",
        "WDAY",
        "XEL",
        "ZM",
        "ZS",
    )

def main():
    year = 2022
    stock_file = f"./data/Nasdaq-100-{year}.xlsx"

    stock_data = pd.read_excel(
        stock_file, header=(0, 1), index_col=0
    )

    tickers = _get_nasdaq100()
    # 某些年份可能某些ticker没有
    for ticker_code in tickers:
        #print(stock_data[ticker_code]["Adj Close"][0])
        if math.isnan(stock_data[ticker_code]["Adj Close"][0]):
            #raise ValueError(f"ticker {ticker_code} not exist.")
            pass
        else:
            print(f"\"{ticker_code}\",")

if __name__ == "__main__":
    main()