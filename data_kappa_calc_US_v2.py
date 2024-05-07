'''
integrating the kappa calculation into the main code
calculating the kappa value for the US stock market, with large number of assets
the number of assets can be derived from the number of valid tickers
'''
import math

from util import PortfolioOptimizer, set_random
from collections import defaultdict
# from data_provider import get_tickers, generate_portfolio_problem_raw_US
import json
import sys
import pickle
import scipy as sp
from util import (
    PortfolioOptimizer,
    calculate_resource,
    cal_cos_similarity,
    set_random,
    construct_data,
)
import numpy as np
from linear_solver import HHLSolver, HybridSolver, NumpySolver, QiskitSolver
import pandas as pd
from qiskit.circuit.library import PhaseEstimation
from qiskit import (
    QuantumCircuit,
    QuantumRegister,
    execute,
    transpile,
    BasicAer,
    ClassicalRegister,
)
# from data_provider import get_tickers_US
from nasdaq100 import get_nasdaq100_by_year
from util import PortfolioOptimizer, set_random

import pandas as pd
import numpy as np
import random

import scipy.optimize as opt
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from typing import List, Tuple, Callable, Union

import pickle


def tool_get_best_s(
    R,
    S,
    P,
):
    """根据(R, S, P)利用优化的方法得到 s

    Args:
        R (_type_): R
        S (_type_): Sigma
        P (_type_): Pi
    """

    def get_condition_number(s):
        global cond
        cond = np.linalg.cond(
            PortfolioOptimizer(
                R,
                S,
                P,
                s[0],
                s[1],
                s[2],
            ).matrix
        )
        return cond

    init_s1 = 1
    init_s2 = 1
    init_s3 = -0.001
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    res = opt.minimize(
        fun=get_condition_number,
        x0=(init_s1, init_s2, init_s3),
        method="Nelder-Mead",
    )
    # print(res)
    return res.x, get_condition_number(res.x), get_condition_number([1, 1, 1])


# num_assets_list = [3, 4, 5, 6, 7...]
def kappa_calc(num_assets_list, stock_file, valid_tickers_path):

    results = []
    benchmark = []
    bench_size = 10  # 10 original
    for num_assets in num_assets_list:
        for _ in range(bench_size):
            tickers = get_tickers_US(
                num_assets=num_assets, valid_tickers_path=valid_tickers_path)
            benchmark.append(tickers)

    for j in range(len(num_assets_list)):
        for i in range(bench_size):
            tickers = benchmark[j*bench_size+i]
            R, S, P = generate_portfolio_problem_raw_US(
                tickers=tickers, stock_file=stock_file, valid_tickers_path=valid_tickers_path
            )
            s_t, kappa_opt, kappa_orig = tool_get_best_s(R, S, P)
            results.append(
                (num_assets_list[j], tickers, s_t, kappa_opt, kappa_orig))
    print(results)
    pd.DataFrame(results, columns=["num_qubits", "tickers", "s1, s2", "kappa_opt", "kappa_original"]).to_csv(
        f"get_data_US/kappa_{sys.argv[1]}_v2.csv",
        index=False,
    )


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


def get_valid_US_tickers(valid_tickers_path: str = "valid_amex_tickers.json"):
    # tickers_path = "C:\\Users\\AC\\Desktop\\Quantum Computation\\TSC\\SAPO-main\\SAPO-main\\get_data_US\\valid_amex_tickers.json"
    tickers_path = valid_tickers_path
    with open(tickers_path) as f:
        ticker_symbols = json.load(f)
    return ticker_symbols


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


if __name__ == "__main__":
    # stock_file = "C:\\Users\\AC\\Desktop\\Quantum Computation\\TSC\\SAPO-main\\SAPO-main\\get_data_US\\nasdaq_data.xlsx"
    # stock_file = "C:\\Users\\AC\\Desktop\\Quantum Computation\\TSC\\SAPO-main\\SAPO-main\\get_data_US\\amex_data.xlsx"
    stock_file_amex = "C:\\Users\\AC\\Desktop\\Quantum Computation\\TSC\\SAPO-main\\SAPO-main\\get_data_US\\amex_data.csv"
    stock_file_nasdaq = "C:\\Users\\AC\\Desktop\\Quantum Computation\\TSC\\SAPO-main\\SAPO-main\\get_data_US\\nasdaq_data.csv"
    stock_file_nyse = "C:\\Users\\AC\\Desktop\\Quantum Computation\\TSC\\SAPO-main\\SAPO-main\\get_data_US\\nyse_data.csv"
    stock_file = stock_file_amex
    valid_amex_tickers_path = "C:\\Users\\AC\\Desktop\\Quantum Computation\\TSC\\SAPO-main\\SAPO-main\\get_data_US\\valid_amex_tickers.json"
    valid_nasdaq_tickers_path = "C:\\Users\\AC\\Desktop\\Quantum Computation\\TSC\\SAPO-main\\SAPO-main\\get_data_US\\valid_nasdaq_tickers.json"
    valid_nyse_tickers_path = "C:\\Users\\AC\\Desktop\\Quantum Computation\\TSC\\SAPO-main\\SAPO-main\\get_data_US\\valid_nyse_tickers.json"

    set_random()
    if sys.argv[1] == "nasdaq":
        valid_tickers_path = valid_nasdaq_tickers_path
        with open(valid_tickers_path) as f:
            ticker_symbols = json.load(f)
        num_assets = len(ticker_symbols)
        # a list with elements from 2 to num_assets
        num_assets_list = list(range(2, num_assets))
        stock_file = stock_file_nasdaq
        kappa_calc(num_assets_list, stock_file, valid_tickers_path)
    elif sys.argv[1] == "amex":
        valid_tickers_path = valid_amex_tickers_path
        with open(valid_tickers_path) as f:
            ticker_symbols = json.load(f)
        num_assets = len(ticker_symbols)
        # a list with elements from 2 to num_assets
        num_assets_list = list(range(2, num_assets))
        stock_file = stock_file_amex
        kappa_calc(num_assets_list, stock_file, valid_tickers_path)
    elif sys.argv[1] == "nyse":
        valid_tickers_path = valid_nyse_tickers_path
        with open(valid_tickers_path) as f:
            ticker_symbols = json.load(f)
        num_assets = len(ticker_symbols)
        # a list with elements from 2 to num_assets
        num_assets_list = list(range(2, num_assets))
        stock_file = stock_file_nyse
        kappa_calc(num_assets_list, stock_file, valid_tickers_path)
    elif sys.argv[1] == "test_amex":
        valid_tickers_path = valid_amex_tickers_path
        with open(valid_tickers_path) as f:
            ticker_symbols = json.load(f)
        num_assets = len(ticker_symbols)
        # a list with elements from 2 to num_assets
        num_assets_list = list(range(num_assets-1, num_assets))
        stock_file = stock_file_amex
        kappa_calc(num_assets_list, stock_file, valid_tickers_path)
