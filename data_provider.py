import numpy as np
import pandas as pd
from typing import List, Tuple, Callable, Union
import math
from nasdaq100 import get_nasdaq100_by_year, get_nasdaq_by_year, get_valid_US_tickers


def generate_hamilton_matrix(matrix_size: int):
    res = np.random.uniform(0, 100, (matrix_size, matrix_size))
    res = np.triu(res)
    res += res.T - np.diag(res.diagonal())
    return res


def get_tickers(
    num_assets: int = 2,
    year: int = 2022,
):
    tickers = np.random.choice(
        get_nasdaq100_by_year(year),
        size=num_assets,
        replace=False,
    ).tolist()
    return tickers


def get_tickers_US(
    num_assets: int = 2,
    valid_tickers_path: str = "valid_amex_tickers.json",
):
    tickers = np.random.choice(
        get_valid_US_tickers(valid_tickers_path),
        # get_nasdaq_by_year(),
        size=num_assets,
        replace=False,
    ).tolist()
    return tickers


def generate_portfolio_problem_raw_US(
    num_assets: int = 2,
    tickers: List[str] = None,
    stock_file: str = "temp",  # ac
    valid_tickers_path: str = "valid_amex_tickers.json",
):
    """
    *_raw: 仅返回最原始的 R, S, P
    """
    if tickers is not None and len(tickers) != 0:
        num_assets = len(tickers)
    elif num_assets < 2:
        raise ValueError("num_assets must be bigger than 1.")
    else:
        tickers = np.random.choice(
            get_valid_US_tickers(valid_tickers_path),
            size=num_assets,
            replace=False,
        ).tolist()

    # read excel take mach time
    # stock_data = pd.read_excel(
    #     stock_file, header=(0, 1), index_col=0
    # )
    stock_data = pd.read_csv(
        stock_file, header=(0, 1), index_col=0
    )

    # 某些年份可能某些 ticker 没有
    # 但, 如果是从 get_nasdaq100_by_year 里面选的, 则不会出现这种情况
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

    return (
        np.mean(period_return, axis=1),  # R
        np.cov(period_return, ddof=1),  # \Sigma
        np.ones(num_assets),  # \Pi
    )


def generate_portfolio_problem_raw_US(
    num_assets: int = 2,
    tickers: List[str] = None,
    stock_file: str = "./data/Nasdaq-100-2022.xlsx",
    valid_tickers_path: str = "valid_amex_tickers.json",
):
    """
    *_raw: 仅返回最原始的 R, S, P
    """
    if tickers is not None and len(tickers) != 0:
        num_assets = len(tickers)
    elif num_assets < 2:
        raise ValueError("num_assets must be bigger than 1.")
    else:
        tickers = np.random.choice(
            get_valid_US_tickers(valid_tickers_path),
            size=num_assets,
            replace=False,
        ).tolist()

    # read excel take mach time
    # stock_data = pd.read_excel(
    #     stock_file, header=(0, 1), index_col=0
    # )
    stock_data = pd.read_csv(
        stock_file,
        # This tells pandas to use the first two rows for the multi-level header
        header=[0, 1],
        index_col=0,     # This uses the first column as the row index
        encoding="ascii"
    )
    # 某些年份可能某些 ticker 没有
    # 但, 如果是从 get_nasdaq100_by_year 里面选的, 则不会出现这种情况
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

    return (
        np.mean(period_return, axis=1),  # R
        np.cov(period_return, ddof=1),  # \Sigma
        np.ones(num_assets),  # \Pi
    )


def generate_portfolio_problem_raw(
    num_assets: int = 2,
    tickers: List[str] = None,
    stock_file: str = "./data/Nasdaq-100-2022.xlsx",
    year: int = 2022,
):
    """
    *_raw: 仅返回最原始的 R, S, P
    """
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

    # 某些年份可能某些 ticker 没有
    # 但, 如果是从 get_nasdaq100_by_year 里面选的, 则不会出现这种情况
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

    return (
        np.mean(period_return, axis=1),  # R
        np.cov(period_return, ddof=1),  # \Sigma
        np.ones(num_assets),  # \Pi
    )


def generate_portfolio_problem(
    num_assets: int = 2, tickers: List[str] = None, scaling: float = 1000.0
):
    """
    原始版本
    """
    if tickers is not None and len(tickers) != 0:
        if num_assets != len(tickers):
            raise ValueError(
                "The assets_num must be the same as the number of tickers."
            )
    elif num_assets < 2:
        raise ValueError("num_assets must be bigger than 1.")
    else:
        tickers = np.random.choice(  # 随机取 num_assets 个 tickers
            _get_nasdaq100(),
            size=num_assets,
            replace=False,
        ).tolist()

    stock_data = pd.read_excel("Nasdaq-100.xlsx", header=(0, 1), index_col=0)
    value_matrix = np.array(
        [stock_data[ticker_code]["Adj Close"] for ticker_code in tickers]
    )
    print(f"{value_matrix}")
    period_return = value_matrix[:, 1:] / value_matrix[:, :-1] - 1
    print(f"Chose tickers: {', '.join(tickers)}.")
    return (
        np.mean(period_return, axis=1),
        np.cov(period_return, ddof=1),
        1 / scaling,  # budget
        np.random.rand(),  # expected_income
        np.ones(num_assets) / scaling,
    )


def _get_nasdaq100() -> Tuple[str]:
    """
    https://en.wikipedia.org/wiki/Nasdaq-100
    https://stockmarketmba.com/stocksinthenasdaq100.php
    """
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


def _download_stock_data(
    get_stocks_function: Callable[[],
                                  Union[Tuple[str], List[str]]] = _get_nasdaq100,
    file_name: str = "Nasdaq-100-test",
) -> None:
    yf.download(
        tickers=get_stocks_function(),
        start="2022-1-19",
        end="2023-4-1",
        # proxy="127.0.0.1:7890",
        group_by="tickers",
    ).to_excel(f"{file_name}.xlsx")


if __name__ == "__main__":
    # _download_stock_data()
    generate_portfolio_problem(
        tickers=["KHC", "AAPL"],
        stock_file="./data/Nasdaq-100-2012.xlsx"
    )
