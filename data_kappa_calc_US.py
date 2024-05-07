from util import PortfolioOptimizer, set_random
from collections import defaultdict
from data_provider import get_tickers, generate_portfolio_problem_raw, generate_portfolio_problem_raw_US
import ray
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
from data_provider import get_tickers_US
from nasdaq100 import get_nasdaq100_by_year
from util import PortfolioOptimizer, set_random

import pandas as pd
import numpy as np
import random

import scipy.optimize as opt
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

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
def kappa_calc(num_assets_list):
    # stock_file = "C:\\Users\\AC\\Desktop\\Quantum Computation\\TSC\\SAPO-main\\SAPO-main\\get_data_US\\nasdaq_data.xlsx"
    # stock_file = "C:\\Users\\AC\\Desktop\\Quantum Computation\\TSC\\SAPO-main\\SAPO-main\\get_data_US\\amex_data.xlsx"
    stock_file = "C:\\Users\\AC\\Desktop\\Quantum Computation\\TSC\\SAPO-main\\SAPO-main\\get_data_US\\amex_data.csv"
    results = []
    benchmark = []
    bench_size = 5  # 10 original
    for num_assets in num_assets_list:
        for _ in range(bench_size):
            tickers = get_tickers_US(num_assets=num_assets)
            benchmark.append(tickers)

    for j in range(len(num_assets_list)):
        for i in range(bench_size):
            tickers = benchmark[j*bench_size+i]
            R, S, P = generate_portfolio_problem_raw_US(
                tickers=tickers, stock_file=stock_file
            )
            s_t, kappa_opt, kappa_orig = tool_get_best_s(R, S, P)
            results.append(
                (num_assets_list[j], tickers, s_t, kappa_opt, kappa_orig))
    print(results)
    pd.DataFrame(results, columns=["num_qubits", "tickers", "s1, s2", "kappa_opt", "kappa_original"]).to_csv(
        f"get_data_US/kappa_v1.csv",
        index=False,
    )


if __name__ == "__main__":
    set_random()
    num_assets_list = [200]
    kappa_calc(num_assets_list)
