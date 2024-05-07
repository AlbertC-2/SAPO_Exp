import numpy as np
import pandas as pd
from typing import List
import math
from nasdaq100 import get_nasdaq100_by_year


def get_tickers(
    num_assets: int = 2,
    year: int = 2012,
):
    tickers = np.random.choice(
                get_nasdaq100_by_year(year),
                size=num_assets,
                replace=False,
            ).tolist()
    return tickers


def generate_portfolio_problem_raw(
    num_assets: int = 2,
    tickers: List[str] = None,
    stock_file: str = "data/Nasdaq-100-2012.xlsx",
    year: int = 2012,
):
    if tickers is not None and len(tickers) != 0:
        num_assets = len(tickers)
    elif num_assets < 2:
        raise ValueError("num_assets must be bigger than 1.")
    else:
        tickers = np.random.choice(
            get_nasdaq100_by_year(year),
            size=num_assets,
            replace=False,
        ).tolist()

    # read excel take mach time
    stock_data = pd.read_excel(
        stock_file, header=(0, 1), index_col=0
    )

    # 某些年份可能某些ticker没有
    for ticker_code in tickers:
        # print(stock_data[ticker_code]["Adj Close"][0])
        if math.isnan(stock_data[ticker_code]["Adj Close"][0]):
            raise ValueError(f"ticker {ticker_code} not exist.")

    # 收盘价
    value_matrix = np.array(
        [stock_data[ticker_code]["Adj Close"] for ticker_code in tickers]
    )
    # 每个标的在当天的收益定义为当天的收盘价相对于前一天收盘价的涨幅
    period_return = value_matrix[:, 1:] / value_matrix[:, :-1] - 1
    # print(f"Chose tickers: {', '.join(tickers)}.")

    return (
        np.mean(period_return, axis=1),  # R
        np.cov(period_return, ddof=1),  # \Sigma
        np.ones(num_assets),  # \Pi
    )


if __name__ == "__main__":
    pass
