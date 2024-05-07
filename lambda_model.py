"""
训练 SVM 模型, 用来预测矩阵的特征值的绝对值的最大值和最小值

result/svm_model_{type}_{num_assets}_tickers_big_scale.pkl
- s = (0.19561288, 0.00044902, -0.00167655)
- matrix_scale = 10000
- year=2022, num_train=3000, num_test=1000,


"""
from data_provider import get_tickers
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


ds_list = []


def main(
    choose_flag=0,
):
    if choose_flag == 0:
        model_lambda(
            num_assets=3,
            year=2022,
            num_train=3000,
            num_test=1000,
            matrix_scale=4000,
            type="max",
        )
    elif choose_flag == 1:
        model_lambda(
            num_assets=3,
            year=2022,
            num_train=3000,
            num_test=1000,
            matrix_scale=4000,
            type="min",
        )
    elif choose_flag == 2:
        for num_assets in np.arange(9, 10, 1):
            print(f"Start testing {num_assets} tickers!")
            model_lambda(
                year=2022,
                num_assets=num_assets,
                num_train=30000,  # 3000, for max is enough
                num_test=1000,
                # matrix_scale=10000,  # 4000,
                matrix_scale=4000,
                type="min",
            )
    elif choose_flag == 3:
        for num_assets in np.arange(2, 7, 1):
            print(f"Start testing {num_assets} tickers!")
            model_lambda(
                year=2022,
                num_assets=num_assets,
                num_train=3000,  # 3000, for max is enough
                num_test=1000,
                matrix_scale=4000,
                type="min",
            )
    elif choose_flag == 4:
        for num_assets in np.arange(2, 7, 1):
            print(f"Start testing {num_assets} tickers!")
            model_lambda(
                year=2022,
                num_assets=num_assets,
                num_train=3000,  # 3000, for max is enough
                num_test=1000,
                matrix_scale=4000,
                type="max",
            )
    elif choose_flag == 5:
        for num_assets in np.arange(2, 7, 1):
            print(f"Start testing {num_assets} tickers!")
            model_lambda(
                year=2022,
                num_assets=num_assets,
                num_train=3000,  # 3000, for max is enough
                num_test=1000,
                matrix_scale=4000,
                type="max",
                s_flag=False,
            )
    elif choose_flag == 6:
        for num_assets in np.arange(2, 7, 1):
            print(f"Start testing {num_assets} tickers!")
            model_lambda(
                year=2022,
                num_assets=num_assets,
                num_train=3000,  # 3000, for max is enough
                num_test=1000,
                matrix_scale=4000,
                type="min",
                s_flag=False,
            )


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
    print(res)
    return res.x


def model_lambda(
    num_assets=2,
    year=2022,
    num_train=3000,
    num_test=1000,
    matrix_scale=4000,
    type="min",  # min 或 max
    s_flag=True,
):
    """|lambda_min or lambda_max| model, use SVM

    Args:
        num_assets (int, optional): _description_. Defaults to 2.
        year (int, optional): _description_. Defaults to 2022.
        num_train (int, optional): _description_. Defaults to 3000.
        num_test (int, optional): _description_. Defaults to 1000.
        matrix_scale (int, optional): _description_. Defaults to 4000.
    """
    if s_flag is False:
        s = (1, 1, 1)
    else:
        s = (0.19561288, 0.00044902, -0.00167655)
    print(f"s={s}")

    def get_data(
        tickers_list,
        type=type,
    ):
        """get data from tickers list

        Args:
            tickers_list (_type_): _description_

        Returns:
            _type_: (LABELS, X)
        """
        lable_list = []
        R_list, S_list = [], []
        # prepare the data
        for t in tickers_list:
            value_matrix = np.array(
                [stock_data[ticker_code]["Adj Close"] for ticker_code in t]
            )
            period_return = value_matrix[:, 1:] / value_matrix[:, :-1] - 1
            R, S, P = (
                np.mean(period_return, axis=1),  # R
                np.cov(period_return, ddof=1),  # \Sigma
                np.ones(num_assets),  # \Pi
            )
            s_matrix = PortfolioOptimizer(R, S, P, *s).matrix * matrix_scale
            eigenvalues = np.linalg.eigvals(s_matrix)
            abs_eigenvalues = np.abs(eigenvalues)
            if type == "min":
                m_abs = np.min(abs_eigenvalues)
            elif type == "max":
                m_abs = np.max(abs_eigenvalues)
            else:
                print("Invalid type!")
            lable_list.append(m_abs)
            R_list.append(R * matrix_scale * s[0])
            S_list.append(S * matrix_scale)
        LABELS = np.array(lable_list)
        R_array = np.array(R_list)
        S_array = np.array(S_list)
        length = int(num_assets + (num_assets**2 + num_assets) // 2)
        X = np.empty((0, length))
        for i in range(len(R_array)):
            S = S_array[i]
            upper_S = S[np.triu_indices(S.shape[0])]  # np.triu 取上三角
            sample = np.concatenate((R_array[i], upper_S))
            X = np.vstack((X, sample))
        return LABELS, X

    stock_file = f"./data/Nasdaq-100-{year}.xlsx"
    # read excel take mach time
    stock_data = pd.read_excel(stock_file, header=(0, 1), index_col=0)

    # 通过 tickers_example 算出 s, 或者使用通用的 s
    tickers_example = get_tickers(num_assets=num_assets, year=year)
    print(f"tickers: {tickers_example}")
    # s = get_best_s(tickers=tickers_example)
    # ds_list.append(s)
    # print(f"s = {s}")

    # get tickers list
    all_ticker_list = get_nasdaq100_by_year(year)
    comb = []
    while len(comb) < (num_train + num_test):
        pair = random.sample(all_ticker_list, num_assets)
        pair.sort()
        if pair not in comb:  # 保证不重复
            comb.append(pair)
    train_tickers_list = comb[0:num_train]
    test_tickers_list = comb[num_train:]
    print(f"len train = {len(train_tickers_list)}")
    print(f"len test = {len(test_tickers_list)}")
    # get training data
    LABELS, X_train = get_data(train_tickers_list)
    print(f"mean min = {np.mean(LABELS)}")
    # SVM model
    if 1:
        """
        C=10.0,
        kernel='rbf',
        gamma='auto',
        epsilon=0.01,
        tol=0.001,
        """
        model = svm.SVR(
            C=3,
            kernel="rbf",
            gamma="auto",
            epsilon=0.01,
            tol=0.001,
        )
        model.fit(X_train, LABELS)

    # Decision Tree model
    if 0:
        model = DecisionTreeRegressor(
            max_depth=8,
            min_samples_split=6,
            min_samples_leaf=3,  # default 1
        )
        model.fit(X_train, LABELS)

    # get testing data
    test_LABELS, X_test = get_data(test_tickers_list)
    # predict and eval
    # training set
    LABELS_predicted = model.predict(X_train)
    l2_loss = mean_squared_error(LABELS, LABELS_predicted)
    rrmse = np.sqrt(l2_loss) / np.mean(LABELS)
    print(f"train l2-loss = {l2_loss}")
    print(f"train RRMSE = {rrmse}")
    # testing set
    test_LABELS_predicted = model.predict(X_test)
    test_l2_loss = mean_squared_error(test_LABELS, test_LABELS_predicted)
    test_rrmse = np.sqrt(test_l2_loss) / np.mean(test_LABELS)
    print(f"test l2-loss = {test_l2_loss}")
    print(f"test RRMSE = {test_rrmse}")

    # model_params = model.get_params()
    # print(f"model_params = {model_params}")

    # train set
    deviations = LABELS_predicted - LABELS  # 计算偏差
    percentages = (deviations / LABELS) * 100  # 计算偏离百分比
    print(
        f"TRAIN: mean = {np.mean(np.abs(percentages))},\
    var = {np.var(np.abs(percentages))},\
    mid = {np.median(np.abs(percentages))}"
    )

    # test set 计算偏差和偏离百分比
    deviations = test_LABELS_predicted - test_LABELS  # 计算偏差
    percentages = (deviations / test_LABELS) * 100  # 计算偏离百分比
    print(
        f"TEST:  mean = {np.mean(np.abs(percentages))},\
    var = {np.var(np.abs(percentages))},\
    mid = {np.median(np.abs(percentages))}"
    )

    # 统计偏大和偏小的个数
    num_overestimated = np.sum(deviations > 0)  # 偏大的个数
    num_underestimated = np.sum(deviations < 0)  # 偏小的个数
    print("偏大的个数:", num_overestimated)
    print("偏小的个数:", num_underestimated)
    print("较好的个数:", len(percentages[np.abs(percentages) < 10]))

    # # 保存数据
    # data = {
    #     "original": test_LABELS,
    #     "predicted": test_LABELS_predicted,
    #     "percentage": percentages,
    # }
    # df = pd.DataFrame(data)
    # """
    #     ds: means different s from different
    #     ss: means same s = (0.19561288, 0.00044902, -0.00167655)
    # """
    # filename = f"result/test_result_{type}_{num_assets}_tickers_ss.csv"
    # df.to_csv(filename, index=False)
    # print(f"finish in {filename}")

    # 保存模型到文件
    if s_flag:
        model_filename = (
            f"result/svm/s/svm_model_{type}_{num_assets}_tickers_ss.pkl"
        )
    else:
        model_filename = (
            f"result/svm/ns/svm_model_{type}_{num_assets}_tickers_ss.pkl"
        )
    with open(model_filename, "wb") as file:
        pickle.dump(model, file)
    print(f"save in {model_filename}")

    if 0:  # 尝试 load 模型并进行预测
        with open(model_filename, "rb") as file:
            svm_model = pickle.load(file)
        test_LABELS_predicted = svm_model.predict([X_test[0]])
        print(test_LABELS_predicted)  # [1.15593399]


if __name__ == "__main__":
    set_random()
    main(choose_flag=3)
    main(choose_flag=4)
