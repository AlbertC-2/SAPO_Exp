from nasdaq100 import get_nasdaq100_by_year
from util import PortfolioOptimizer
from data_provider import generate_portfolio_problem_raw, get_tickers
import pandas as pd
import numpy as np
from itertools import combinations
import scipy.optimize as opt


def main():
    test_flag = 2

    if test_flag == 0:
        temp_test()
    elif test_flag == 1:
        test_pearson_corr()
    elif test_flag == 2:
        test_s_temp()
    elif test_flag == 3:
        test_all_use_one_s()
    elif test_flag == 4:
        pass


def test_s_temp():
    num_assets = 2
    year = 2012
    stock_file = f"./data/Nasdaq-100-{year}.xlsx"
    tickers = get_tickers(num_assets=num_assets, year=year)
    tickers = ["ADSK", "KLAC"]
    print(f"tickers: {tickers}")
    s = _get_best_scarling(tickers, stock_file=stock_file)
    print(f"final s = {s}")


def _get_best_scarling(
    tickers,
    stock_file,
):
    np.set_printoptions(linewidth=np.inf)

    cost_values = []
    R, S, P = generate_portfolio_problem_raw(
        tickers=tickers, stock_file=stock_file
    )
    original_matrix = PortfolioOptimizer(R, S, P, 1, 1, 1).matrix
    original_cond = np.linalg.cond(original_matrix)
    print(f"original_cond = {original_cond}")
    print(f"original_matrix = \n{original_matrix}")

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

    def log_cost(x):
        cost_values.append(cond)

    init_s1 = 1
    init_s2 = 0
    init_s3 = -0.001
    print(f"init s = {init_s1}, {init_s2}, {init_s3}")
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    res = opt.minimize(
        get_condition_number,
        (init_s1, init_s2, init_s3),
        options={"maxiter": 20},  # 50
        callback=log_cost,
        method="Nelder-Mead",  # method='BFGS'
    )
    """
    Powell 非常power
    """
    print(res)

    optimized_matrix = PortfolioOptimizer(R, S, P, *(res.x)).matrix
    optimized_cond = np.linalg.cond(optimized_matrix)
    print(f"log_cost = {cost_values}")
    print(f"optimized_cond = {optimized_cond}")

    return res.x


def test_all_use_one_s(
    s=(3.66386350e-01, 1.59224564e-04, -5.40439030e-04),
    # s=(-0.1077046, -0.00016113, -0.00098139),
    num_combinations=100,
):
    def get_best_scarling(
        R,
        S,
        P,
    ):
        def get_condition_number(s):
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
            get_condition_number,
            (init_s1, init_s2, init_s3),
            options={"maxiter": 50},
            method="BFGS",
        )
        return res.x

    num_assets_list = []
    year_list = []
    combination_list = []
    original_cond_list = []
    s_cond_list = []
    optimized_cond_list = []

    for year in np.arange(2012, 2023, 1):
        print(f"year: {year}")
        stock_file = f"./data/Nasdaq-100-{year}.xlsx"
        stock_data = pd.read_excel(stock_file, header=(0, 1), index_col=0)
        for num_assets in range(2, 8, 1):
            for i in range(num_combinations):
                tickers = np.random.choice(
                    get_nasdaq100_by_year(year),
                    size=num_assets,
                    replace=False,
                ).tolist()
                value_matrix = np.array(
                    [
                        stock_data[ticker_code]["Adj Close"]
                        for ticker_code in tickers
                    ]
                )
                period_return = value_matrix[:, 1:] / value_matrix[:, :-1] - 1
                R, S, P = (
                    np.mean(period_return, axis=1),
                    np.cov(period_return, ddof=1),
                    np.ones(num_assets),
                )
                original_matrix = PortfolioOptimizer(R, S, P, 1, 1, 1).matrix
                original_cond = np.linalg.cond(original_matrix)
                s_matrix = PortfolioOptimizer(R, S, P, *s).matrix
                s_cond = np.linalg.cond(s_matrix)
                best_s = get_best_scarling(R, S, P)
                optimized_matrix = PortfolioOptimizer(R, S, P, *best_s).matrix
                optimized_cond = np.linalg.cond(optimized_matrix)

                num_assets_list.append(num_assets)
                year_list.append(year)
                combination_list.append(i)
                original_cond_list.append(original_cond)
                s_cond_list.append(s_cond)
                optimized_cond_list.append(optimized_cond)

                # with open("output.txt", "a") as file:
                #     print(f"For num_assets={num_assets} | In {year} year | Combination {i} - {tickers}: original_cond = {original_cond}; optimized_cond = {optimized_cond}",
                #           file=file)

    file_name = "s_output_all.csv"
    data = {
        "num_assets": num_assets_list,
        "year": year_list,
        "comb": combination_list,
        "original_cond": original_cond_list,
        "s_cond": s_cond_list,
        "optimized_cond": optimized_cond_list,
    }
    df = pd.DataFrame(data)
    df.to_csv(file_name, index=False)
    print("finish csv")


def cosine_similarity_matrix(
    matrix1,
    matrix2,
):
    vector1, vector2 = matrix1.flatten(), matrix2.flatten()
    r = cosine_similarity(vector1, vector2)
    return r


def cosine_similarity(
    vector1,
    vector2,
):
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    similarity = dot_product / (norm1 * norm2)
    return similarity


def pearson_corr(
    matrix1,
    matrix2,
):
    # 计算标准差
    std1 = np.std(matrix1)
    std2 = np.std(matrix2)
    # 计算协方差矩阵
    cov_matrix = np.cov(matrix1.flatten(), matrix2.flatten(), bias=True)
    # 提取协方差值
    cov = cov_matrix[0, 1]
    # 计算皮尔逊相关系数
    r = cov / (std1 * std2)
    return r


def test_pearson_corr():
    def get_matrix(
        tickers,
    ):
        value_matrix = np.array(
            [stock_data[ticker_code]["Adj Close"] for ticker_code in tickers]
        )
        period_return = value_matrix[:, 1:] / value_matrix[:, :-1] - 1
        R, S, P = (
            np.mean(period_return, axis=1),  # R
            np.cov(period_return, ddof=1),  # \Sigma
            np.ones(num_assets),  # \Pi
        )
        matrix = PortfolioOptimizer(R, S, P, 1, 1, 1).matrix
        return matrix

    year = 2012
    stock_file = f"./data/Nasdaq-100-{year}.xlsx"
    # read excel take mach time
    stock_data = pd.read_excel(stock_file, header=(0, 1), index_col=0)

    num_assets_list = []
    max_values = []
    min_values = []
    mean_values = []
    medians = []

    for num_assets in np.arange(2, 10, 1):
        tickers = get_tickers(num_assets=num_assets, year=year)
        original_matrix = get_matrix(tickers)
        num_test = 1000  # 测试一部分
        r_list = []
        for i in range(num_test):
            tickers_temp = get_tickers(num_assets=num_assets, year=year)
            original_matrix_temp = get_matrix(tickers_temp)
            r = pearson_corr(original_matrix, original_matrix_temp)
            # print(f"num_assets = {num_assets}, r = {r}")
            r_list.append(r)
        max_value = np.max(r_list)
        min_value = np.min(r_list)
        mean_value = np.mean(r_list)
        median = np.median(r_list)

        num_assets_list.append(num_assets)
        max_values.append(max_value)
        min_values.append(min_value)
        mean_values.append(mean_value)
        medians.append(median)

    data = {
        "num_assets": num_assets_list,
        "max_value": max_values,
        "min_value": min_values,
        "mean_value": mean_values,
        "median": medians,
    }
    df = pd.DataFrame(data)
    df.to_csv("pearson_corr_more.csv", index=False)
    print("finish csv")


def temp_test():
    num_assets = 2
    year = 2012
    stock_file = f"./data/Nasdaq-100-{year}.xlsx"
    # read excel take mach time
    stock_data = pd.read_excel(stock_file, header=(0, 1), index_col=0)

    tickers = get_tickers(num_assets=num_assets, year=year)
    print(f"tickers: {tickers}")
    value_matrix = np.array(
        [stock_data[ticker_code]["Adj Close"] for ticker_code in tickers]
    )

    period_return = value_matrix[:, 1:] / value_matrix[:, :-1] - 1

    R, S, P = (
        np.mean(period_return, axis=1),  # R
        np.cov(period_return, ddof=1),  # \Sigma
        np.ones(num_assets),  # \Pi
    )

    original_matrix = PortfolioOptimizer(R, S, P, 1, 1, 1).matrix

    print(original_matrix)

    for n in np.arange(2, 7, 1):
        tickers = get_tickers(num_assets=n, year=year)
        value_matrix = np.array(
            [stock_data[ticker_code]["Adj Close"] for ticker_code in tickers]
        )

        period_return = value_matrix[:, 1:] / value_matrix[:, :-1] - 1

        R, S, P = (
            np.mean(period_return, axis=1),  # R
            np.cov(period_return, ddof=1),  # \Sigma
            np.ones(n),  # \Pi
        )

        original_matrix_temp = PortfolioOptimizer(R, S, P, 1, 1, 1).matrix
        r = cosine_similarity_matrix(original_matrix, original_matrix_temp)
        print(f"n = {n}, r = {r}")

    if 0:
        all_ticker_list = get_nasdaq100_by_year(year)

        combinations_list = list(combinations(all_ticker_list, num_assets))

        print(f"len of combinations_list = {len(combinations_list)}")
        if len(combinations_list) > 10000:
            combinations_list = combinations_list[0:10000]

        pearson_corrs = []
        for tickers in combinations_list:
            # print(combination)
            value_matrix = np.array(
                [
                    stock_data[ticker_code]["Adj Close"]
                    for ticker_code in tickers
                ]
            )

            period_return = value_matrix[:, 1:] / value_matrix[:, :-1] - 1

            R, S, P = (
                np.mean(period_return, axis=1),  # R
                np.cov(period_return, ddof=1),  # \Sigma
                np.ones(num_assets),  # \Pi
            )

            original_matrix_temp = PortfolioOptimizer(R, S, P, 1, 1, 1).matrix
            # r = pearson_corr(original_matrix, original_matrix_temp)
            r = cosine_similarity_matrix(original_matrix, original_matrix_temp)
            print(r)
            pearson_corrs.append(r)

    if 0:
        all_ticker_list = get_nasdaq100_by_year(year)

        combinations_list = list(combinations(all_ticker_list, num_assets))

        print(f"len of combinations_list = {len(combinations_list)}")
        if len(combinations_list) > 10000:
            combinations_list = combinations_list[0:10000]

        for tickers in combinations_list:
            # print(combination)
            value_matrix = np.array(
                [
                    stock_data[ticker_code]["Adj Close"]
                    for ticker_code in tickers
                ]
            )

            period_return = value_matrix[:, 1:] / value_matrix[:, :-1] - 1

            R, S, P = (
                np.mean(period_return, axis=1),  # R
                np.cov(period_return, ddof=1),  # \Sigma
                np.ones(num_assets),  # \Pi
            )

            original_matrix = PortfolioOptimizer(R, S, P, 1, 1, 1).matrix


if __name__ == "__main__":
    main()
