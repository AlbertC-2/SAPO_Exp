from collections import defaultdict
from data_provider import get_tickers, generate_portfolio_problem_raw
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


# NOTE：编号1
def qiskit_acc_fixed_qubit(num_qubit):
    # HINT: 固定比特数下qiskit精度（取5就行）
    stock_file = "./data/Nasdaq-100-2022.xlsx"
    results = []
    benchmark = defaultdict(list)
    bench_size = 20
    for num_assets in range(2, 7, 1):
        for i in range(bench_size):
            benchmark[num_assets].append(get_tickers(num_assets, 2022))

    for num_assets in range(2, 7, 1):
        for i in range(bench_size):
            print(f"[num_assets={num_assets}] iter{i}")
            tickers = benchmark[num_assets][i]
            R, S, P = generate_portfolio_problem_raw(
                tickers=tickers, stock_file=stock_file
            )
            A, b = PortfolioOptimizer(R, S, P).equation
            std_res = NumpySolver(A, b).solve()[2: 2 + num_assets]
            qiskit_res = QiskitSolver(A, b, num_qubit).solve()[
                2: 2 + num_assets
            ]
            results.append(
                (num_assets, tickers, cal_cos_similarity(std_res, qiskit_res))
            )
    pd.DataFrame(results, columns=["num_assets", "tickers", "acc"]).to_csv(
        f"result/main/fix-qubit/qiskit-{num_qubit}.csv", index=False
    )


# NOTE: 编号2
def qiskit_acc_fixed_stock(num_assets, s, e):
    # HINT: 固定股票数下qiskit精度
    stock_file = "./data/Nasdaq-100-2022.xlsx"
    results = []
    benchmark = []
    bench_size = 20
    for _ in range(bench_size):
        tickers = get_tickers(num_assets=num_assets, year=2022)
        benchmark.append(tickers)

    for num_qubits in range(s, e + 1, 2):
        for i in range(bench_size):
            print(f"[num_qubits={num_qubits}] iter{i}")
            tickers = benchmark[i]
            R, S, P = generate_portfolio_problem_raw(
                tickers=tickers, stock_file=stock_file
            )
            A, b = PortfolioOptimizer(R, S, P).equation
            std_res = NumpySolver(A, b).solve()[2: 2 + num_assets]
            qiskit_res = QiskitSolver(A, b, num_qubits).solve()[
                2: 2 + num_assets
            ]
            results.append(
                (num_qubits, tickers, cal_cos_similarity(std_res, qiskit_res))
            )
    pd.DataFrame(results, columns=["num_qubits", "tickers", "acc"]).to_csv(
        f"result/main/fix-stock/qiskit-{num_assets}-{s}to{e}.csv", index=False
    )


# NOTE: 编号3
def hybrid_acc_fixed_qubit(num_qubit):
    # HINT: 固定比特数下qiskit精度（取5就行）
    stock_file = "./data/Nasdaq-100-2022.xlsx"
    results = []
    benchmark = defaultdict(list)
    bench_size = 20
    for num_assets in range(2, 7, 1):
        for i in range(bench_size):
            benchmark[num_assets].append(get_tickers(num_assets, 2022))

    def qpe_estimate(A, b):
        vector_reg = QuantumRegister(int(np.log2(np.shape(b))))
        vector_circuit = QuantumCircuit(vector_reg)
        vector_circuit.iso(b, vector_circuit.qubits, None)
        matrix_circuit = QuantumCircuit(vector_reg.size)
        matrix_circuit.unitary(
            sp.linalg.expm(1j * A * np.pi), matrix_circuit.qubits
        )
        qpe_reg = QuantumRegister(num_qubit)
        qpe_circuit = PhaseEstimation(qpe_reg.size, matrix_circuit)
        circuit = QuantumCircuit(vector_reg, qpe_reg)
        measure_reg = ClassicalRegister(qpe_reg.size)
        circuit = QuantumCircuit(vector_reg, qpe_reg, measure_reg)
        circuit.append(vector_circuit, vector_reg[:])
        circuit.append(qpe_circuit, qpe_reg[:] + vector_reg[:])
        circuit.measure(qpe_reg[:], measure_reg[:])
        counts = (
            execute(
                circuit, BasicAer.get_backend("qasm_simulator"), shots=1000
            )
            .result()
            .get_counts()
        )
        result = []
        for state, count in counts.items():
            result.append((int(state[::-1], 2) / (2**num_qubit), count))
        result.sort(key=lambda x: x[1], reverse=True)
        lambdas = []
        for i in range(min(int(np.shape(b)[0]), len(result))):
            phi = result[i][0]
            if phi < 0.5:
                lambdas.append(2 * phi)
            else:
                lambdas.append(2 * (phi - 1))
        lambdas = np.abs(lambdas)
        return np.min(lambdas), np.max(lambdas)

    # HINT: 固定比特数下hybird精度
    stock_file = "./data/Nasdaq-100-2022.xlsx"
    results = []
    for num_assets in range(2, 7, 1):
        for i in range(bench_size):
            print(f"[num_assets={num_assets}] iter{i}")
            tickers = benchmark[num_assets][i]
            R, S, P = generate_portfolio_problem_raw(
                tickers=tickers, stock_file=stock_file
            )
            A, b = PortfolioOptimizer(R, S, P).equation
            std_res = NumpySolver(A, b).solve()[2: 2 + num_assets]
            try:
                x, y = qpe_estimate(A / np.trace(A), b / np.linalg.norm(b))
            except Exception as e:
                print(e)
                results.append((num_assets, [], np.nan))
                continue
            hybrid_res = HybridSolver(
                A, b, lambda_min=x, lambda_max=y, phase_qubit_num=num_qubit
            ).solve()[2: 2 + num_assets]
            temp_res = (
                num_assets,
                tickers,
                cal_cos_similarity(std_res, hybrid_res),
            )
            print(temp_res)
            results.append(temp_res)
    pd.DataFrame(results, columns=["num_assets", "tickers", "acc"]).to_csv(
        f"result/main/fix-qubit/hybrid-{num_qubit}.csv", index=False
    )


# NOTE: 编号4
def hybrid_acc_fixed_stock(num_assets, s, e):
    stock_file = "./data/Nasdaq-100-2022.xlsx"
    results = []
    benchmark = []
    bench_size = 20
    for i in range(bench_size):
        benchmark.append(get_tickers(num_assets, 2022))

    def qpe_estimate(A, b, num_qubit):
        vector_reg = QuantumRegister(int(np.log2(np.shape(b))))
        vector_circuit = QuantumCircuit(vector_reg)
        vector_circuit.iso(b, vector_circuit.qubits, None)
        matrix_circuit = QuantumCircuit(vector_reg.size)
        matrix_circuit.unitary(
            sp.linalg.expm(1j * A * np.pi), matrix_circuit.qubits
        )
        qpe_reg = QuantumRegister(num_qubit)
        qpe_circuit = PhaseEstimation(qpe_reg.size, matrix_circuit)
        circuit = QuantumCircuit(vector_reg, qpe_reg)
        measure_reg = ClassicalRegister(qpe_reg.size)
        circuit = QuantumCircuit(vector_reg, qpe_reg, measure_reg)
        circuit.append(vector_circuit, vector_reg[:])
        circuit.append(qpe_circuit, qpe_reg[:] + vector_reg[:])
        circuit.measure(qpe_reg[:], measure_reg[:])
        counts = (
            execute(
                circuit, BasicAer.get_backend("qasm_simulator"), shots=1000
            )
            .result()
            .get_counts()
        )
        result = []
        for state, count in counts.items():
            result.append((int(state[::-1], 2) / (2**num_qubit), count))
        result.sort(key=lambda x: x[1], reverse=True)
        lambdas = []
        for i in range(min(int(np.shape(b)[0]), len(result))):
            phi = result[i][0]
            if phi < 0.5:
                lambdas.append(2 * phi)
            else:
                lambdas.append(2 * (phi - 1))
        lambdas = np.abs(lambdas)
        return np.min(lambdas), np.max(lambdas)

    # HINT: 固定资产数下hybird精度
    stock_file = "./data/Nasdaq-100-2022.xlsx"
    results = []
    for num_qubits in range(s, e + 1, 2):
        for i in range(bench_size):
            print(f"[num_qubits={num_qubits}] iter{i}")
            tickers = benchmark[i]
            R, S, P = generate_portfolio_problem_raw(
                tickers=tickers, stock_file=stock_file
            )
            A, b = PortfolioOptimizer(R, S, P).equation
            std_res = NumpySolver(A, b).solve()[2: 2 + num_assets]
            try:
                x, y = qpe_estimate(
                    A / np.trace(A), b / np.linalg.norm(b), num_qubits
                )
            except Exception as ex:
                print(ex)
                results.append((num_qubits, [], np.nan))
                continue
            hybrid_res = HybridSolver(
                A, b, lambda_min=x, lambda_max=y, phase_qubit_num=num_qubits
            ).solve()[2: 2 + num_assets]
            temp_res = (
                num_qubits,
                tickers,
                cal_cos_similarity(std_res, hybrid_res),
            )
            print(temp_res)
            results.append(temp_res)
    pd.DataFrame(results, columns=["num_qubits", "tickers", "acc"]).to_csv(
        f"result/main/fix-stock/hybrid-{num_assets}-{s}to{e}.csv", index=False
    )


# FIXME: 一次性跑一堆服务器要寄了
# def qiskit_acc_fixed_stock(num_assets, s, e):
#     # HINT: 固定股票数下qiskit精度
#     stock_file = "./data/Nasdaq-100-2022.xlsx"
#     results = []
#     benchmark = []
#     benchmark_size = 100
#     for _ in range(benchmark_size):
#         tickers = get_tickers(num_assets=num_assets, year=2022)
#         benchmark.append(tickers)
#     print(benchmark)

#     @ray.remote
#     def run(num_assets, num_qubits, tickers, i):
#         print(f"[num_qubits={num_qubits}] iter{i}")
#         R, S, P = generate_portfolio_problem_raw(
#             tickers=tickers, stock_file=stock_file
#         )
#         A, b = PortfolioOptimizer(R, S, P).equation
#         std_res = NumpySolver(A, b).solve()[2 : 2 + num_assets]
#         qiskit_res = QiskitSolver(A, b, num_qubits).solve()[2 : 2 + num_assets]
#         return (num_qubits, tickers, cal_cos_similarity(std_res, qiskit_res))

#     for num_qubits in range(s, e, 2):
#         results.extend(
#             ray.get(
#                 [
#                     run.remote(num_assets, num_qubits, benchmark[i], i)
#                     for i in range(benchmark_size)
#                 ]
#             )
#         )


#     pd.DataFrame(results, columns=["num_qubits", "tickers", "acc"]).to_csv(
#         f"result/main/fix-stock/qiskit-{num_assets}-{s}to{e}.csv", index=False
#     )
# NOTE：编号5
def sapo_without_scale_fix_qubit(num_qubit):
    # HINT: 固定比特数下不适用scale的精度（取5就行）
    stock_file = "./data/Nasdaq-100-2022.xlsx"
    results = []
    benchmark = defaultdict(list)
    matrix_scale = 4000
    bench_size = 20
    s = (1, 1, 1)
    for num_assets in range(2, 7, 1):
        for i in range(bench_size):
            benchmark[num_assets].append(get_tickers(num_assets, 2022))

    for num_assets in range(2, 7, 1):
        for i in range(bench_size):
            print(f"[num_assets={num_assets}] iter{i}")
            tickers = benchmark[num_assets][i]
            R, S, P = generate_portfolio_problem_raw(
                tickers=tickers, stock_file=stock_file
            )
            A, b = PortfolioOptimizer(R, S, P).equation
            std_res = NumpySolver(A, b).solve()[2: 2 + num_assets]
            for type in ["min", "max"]:
                model_filename = f"result/svm/ns/svm_model_{type}_{num_assets}_tickers_ss.pkl"
                with open(model_filename, "rb") as file:
                    svm_model = pickle.load(file)
                X = construct_data(num_assets, R, S, matrix_scale, s)
                lambda_m = svm_model.predict(X)
                if type == "min":
                    # np.abs: SVR 的输出可能为负数, 可以考虑增加 matrix_scale
                    # 使得 label 的值远离0点
                    lambda_min = np.abs(lambda_m[0] / matrix_scale)
                elif type == "max":
                    lambda_max = np.abs(lambda_m[0] / matrix_scale)

            sapo_res = HHLSolver(
                A, b, lambda_min, lambda_max, phase_qubit_num=num_qubit
            ).solve()[2: 2 + num_assets]
            results.append(
                (num_assets, tickers, cal_cos_similarity(std_res, sapo_res))
            )
    pd.DataFrame(results, columns=["num_assets", "tickers", "acc"]).to_csv(
        f"result/main/fix-qubit/sapo_without_scale-{num_qubit}.csv",
        index=False,
    )


# NOTE: 编号6
def sapo_without_scale_fix_stock(num_assets, st, ed):
    # HINT: 固定股票数下不使用scale的精度
    stock_file = "./data/Nasdaq-100-2022.xlsx"
    results = []
    matrix_scale = 4000
    benchmark = []
    bench_size = 20
    for _ in range(bench_size):
        tickers = get_tickers(num_assets=num_assets, year=2022)
        benchmark.append(tickers)

    for num_qubits in range(st, ed + 1, 2):
        for i in range(bench_size):
            print(f"[num_qubits={num_qubits}] iter{i}")
            tickers = benchmark[i]
            R, S, P = generate_portfolio_problem_raw(
                tickers=tickers, stock_file=stock_file
            )
            A, b = PortfolioOptimizer(R, S, P).equation
            std_res = NumpySolver(A, b).solve()[2: 2 + num_assets]
            for type in ["min", "max"]:
                model_filename = f"result/svm/ns/svm_model_{type}_{num_assets}_tickers_ss.pkl"
                with open(model_filename, "rb") as file:
                    svm_model = pickle.load(file)
                X = construct_data(num_assets, R, S, matrix_scale, (1, 1, 1))
                lambda_m = svm_model.predict(X)
                if type == "min":
                    # np.abs: SVR 的输出可能为负数, 可以考虑增加 matrix_scale
                    # 使得 label 的值远离0点
                    lambda_min = np.abs(lambda_m[0] / matrix_scale)
                elif type == "max":
                    lambda_max = np.abs(lambda_m[0] / matrix_scale)

            sapo_res = HHLSolver(
                A, b, lambda_min, lambda_max, phase_qubit_num=num_qubits
            ).solve()[2: 2 + num_assets]
            results.append(
                (num_qubits, tickers, cal_cos_similarity(std_res, sapo_res))
            )
    pd.DataFrame(results, columns=["num_qubits", "tickers", "acc"]).to_csv(
        f"result/main/fix-stock/sapo_without_scale-{num_assets}-{st}to{ed}.csv",
        index=False,
    )


# NOTE: 编号7
def sapo_without_eigen_fix_qubit(num_qubit):
    # HINT: 固定比特数下不适用eigen精度（取5就行）
    stock_file = "./data/Nasdaq-100-2022.xlsx"
    results = []
    s = (0.19561288, 0.00044902, -0.00167655)  # ac: 从何而来?
    benchmark = defaultdict(list)
    bench_size = 20
    for num_assets in range(2, 7, 1):
        for i in range(bench_size):
            benchmark[num_assets].append(get_tickers(num_assets, 2022))

    for num_assets in range(2, 7, 1):
        for i in range(bench_size):
            print(f"[num_assets={num_assets}] iter{i}")
            tickers = benchmark[num_assets][i]
            R, S, P = generate_portfolio_problem_raw(
                tickers=tickers, stock_file=stock_file
            )
            A, b = PortfolioOptimizer(R, S, P, *s).equation
            std_res = NumpySolver(A, b).solve()[2: 2 + num_assets]
            sapo_res = QiskitSolver(A, b, num_qubit).solve()[
                2: 2 + num_assets
            ]
            results.append(
                (num_assets, tickers, cal_cos_similarity(std_res, sapo_res))
            )
    pd.DataFrame(results, columns=["num_assets", "tickers", "acc"]).to_csv(
        f"result/main/fix-qubit/sapo_without_eigen-{num_qubit}.csv",
        index=False,
    )


# NOTE: 编号8
def sapo_without_eigen_fix_stock(num_assets, st, ed):
    # HINT: 固定股票数下qiskit精度
    stock_file = "./data/Nasdaq-100-2022.xlsx"
    results = []
    benchmark = []
    bench_size = 20
    s = (0.19561288, 0.00044902, -0.00167655)
    for _ in range(bench_size):
        tickers = get_tickers(num_assets=num_assets, year=2022)
        benchmark.append(tickers)

    for num_qubits in range(st, ed + 1, 2):
        for i in range(bench_size):
            print(f"[num_qubits={num_qubits}] iter{i}")
            tickers = benchmark[i]
            R, S, P = generate_portfolio_problem_raw(
                tickers=tickers, stock_file=stock_file
            )
            A, b = PortfolioOptimizer(R, S, P, *s).equation
            std_res = NumpySolver(A, b).solve()[2: 2 + num_assets]
            sapo_res = QiskitSolver(A, b, num_qubits).solve()[
                2: 2 + num_assets
            ]
            results.append(
                (num_qubits, tickers, cal_cos_similarity(std_res, sapo_res))
            )
    pd.DataFrame(results, columns=["num_qubits", "tickers", "acc"]).to_csv(
        f"result/main/fix-stock/sapo_without_eigen-{num_assets}-{st}to{ed}.csv",
        index=False,
    )


# NOTE: 编号9
def sapo_fix_qubit(num_qubit):
    # HINT: 固定比特数下SAPO的精度（取5就行）
    stock_file = "./data/Nasdaq-100-2022.xlsx"
    results = []
    benchmark = defaultdict(list)
    matrix_scale = 4000
    bench_size = 20
    s = (0.19561288, 0.00044902, -0.00167655)
    for num_assets in range(2, 7, 1):
        for i in range(bench_size):
            benchmark[num_assets].append(get_tickers(num_assets, 2022))

    for num_assets in range(2, 7, 1):
        for i in range(bench_size):
            print(f"[num_assets={num_assets}] iter{i}")
            tickers = benchmark[num_assets][i]
            R, S, P = generate_portfolio_problem_raw(
                tickers=tickers, stock_file=stock_file
            )
            A, b = PortfolioOptimizer(R, S, P, *s).equation
            std_res = NumpySolver(A, b).solve()[2: 2 + num_assets]
            for type in ["min", "max"]:
                model_filename = f"result/svm/s/svm_model_{type}_{num_assets}_tickers_ss.pkl"
                with open(model_filename, "rb") as file:
                    svm_model = pickle.load(file)
                X = construct_data(num_assets, R, S, matrix_scale, s)
                lambda_m = svm_model.predict(X)
                if type == "min":
                    # np.abs: SVR 的输出可能为负数, 可以考虑增加 matrix_scale
                    # 使得 label 的值远离0点
                    lambda_min = np.abs(lambda_m[0] / matrix_scale)
                elif type == "max":
                    lambda_max = np.abs(lambda_m[0] / matrix_scale)

            sapo_res = HHLSolver(
                A, b, lambda_min, lambda_max, phase_qubit_num=num_qubit
            ).solve()[2: 2 + num_assets]
            results.append(
                (num_assets, tickers, cal_cos_similarity(std_res, sapo_res))
            )
    pd.DataFrame(results, columns=["num_assets", "tickers", "acc"]).to_csv(
        f"result/main/fix-qubit/sapo-{num_qubit}.csv",
        index=False,
    )


# NOTE: 编号10
def sapo_fix_stock(num_assets, st, ed):
    # HINT: 固定股票数下不使用scale的精度
    stock_file = "./data/Nasdaq-100-2022.xlsx"
    results = []
    matrix_scale = 4000
    s = (0.19561288, 0.00044902, -0.00167655)
    benchmark = []
    bench_size = 20
    for _ in range(bench_size):
        tickers = get_tickers(num_assets=num_assets, year=2022)
        benchmark.append(tickers)

    for num_qubits in range(st, ed + 1, 2):
        for i in range(bench_size):
            print(f"[num_qubits={num_qubits}] iter{i}")
            tickers = benchmark[i]
            R, S, P = generate_portfolio_problem_raw(
                tickers=tickers, stock_file=stock_file
            )
            A, b = PortfolioOptimizer(R, S, P, *s).equation
            std_res = NumpySolver(A, b).solve()[2: 2 + num_assets]
            for type in ["min", "max"]:
                model_filename = f"result/svm/s/svm_model_{type}_{num_assets}_tickers_ss.pkl"
                with open(model_filename, "rb") as file:
                    svm_model = pickle.load(file)
                X = construct_data(num_assets, R, S, matrix_scale, (1, 1, 1))
                lambda_m = svm_model.predict(X)
                if type == "min":
                    # np.abs: SVR 的输出可能为负数, 可以考虑增加 matrix_scale
                    # 使得 label 的值远离0点
                    lambda_min = np.abs(lambda_m[0] / matrix_scale)
                elif type == "max":
                    lambda_max = np.abs(lambda_m[0] / matrix_scale)

            sapo_res = HHLSolver(
                A, b, lambda_min, lambda_max, phase_qubit_num=num_qubits
            ).solve()[2: 2 + num_assets]
            results.append(
                (num_qubits, tickers, cal_cos_similarity(std_res, sapo_res))
            )
    pd.DataFrame(results, columns=["num_qubits", "tickers", "acc"]).to_csv(
        f"result/main/fix-stock/sapo-{num_assets}-{st}to{ed}.csv",
        index=False,
    )


def test_example():
    matrix_scale = 4000
    s = (0.19561288, 0.00044902, -0.00167655)

    def construct_data(
        R,
        S,
    ):
        length = int(num_assets + (num_assets**2 + num_assets) // 2)
        X = np.empty((0, length))
        R_array = R * matrix_scale * s[0]
        upper_S = S[np.triu_indices(S.shape[0])]
        sample = np.concatenate((R_array, upper_S))
        X = np.vstack((X, sample))
        return X

    @ray.remote
    def run_test_iter(n, e):
        tickers = get_tickers(num_assets=n, year=2022)
        stock_file = f"./data/Nasdaq-100-2022.xlsx"
        R, S, P = generate_portfolio_problem_raw(
            tickers=tickers, stock_file=stock_file
        )
        optimized_matrix, b = PortfolioOptimizer(R, S, P, *s).equation
        for type in ["min", "max"]:
            model_filename = f"result/svm/svm_model_{type}_{n}_tickers_ss.pkl"

            with open(model_filename, "rb") as file:
                svm_model = pickle.load(file)
            X = construct_data(R, S)
            lambda_m = svm_model.predict(X)
            if type == "min":
                # np.abs: SVR 的输出可能为负数, 可以考虑增加 matrix_scale
                # 使得 label 的值远离0点
                lambda_min = np.abs(lambda_m[0] / matrix_scale)
            elif type == "max":
                lambda_max = np.abs(lambda_m[0] / matrix_scale)

        A = np.array(optimized_matrix, dtype=float)
        classic_result = NumpySolver(A, b).solve()[2: 2 + num_assets]
        my_solver = HHLSolver(A, b, lambda_min, lambda_max, epsilon=e)
        my_circ = my_solver.construct_circuit()
        qiskit_solver = QiskitSolver(A, b)
        qiskit_circ = qiskit_solver.construct_circuit()
        trans_qiskit_circ: QuantumCircuit = transpile(
            qiskit_circ, backend=BasicAer.get_backend("qasm_simulator")
        )
        return [
            cal_cos_similarity(
                classic_result, my_solver.solve(my_circ)[2: 2 + num_assets]
            ),
            my_circ.num_qubits,
            calculate_resource(
                my_solver.phase_qubit_num, my_solver.vector_qubit_num, "CNOT"
            ),
            calculate_resource(
                my_solver.phase_qubit_num, my_solver.vector_qubit_num, "Depth"
            ),
            cal_cos_similarity(
                classic_result,
                qiskit_solver.solve(qiskit_circ)[2: 2 + num_assets],
            ),
            qiskit_circ.num_qubits,
            trans_qiskit_circ.count_ops()["cx"],
            trans_qiskit_circ.depth(),
        ]

    result = []
    for num_assets in range(2, 7, 1):
        for epsilon in [2, 4, 6, 8, 16, 32]:
            iter_res = [
                run_test_iter.remote(num_assets, 1 / epsilon)
                for _ in range(20)
            ]
            iter_res = np.array(ray.get(iter_res))
            my_acc = iter_res[:, 0].mean()
            my_qubits = iter_res[:, 1].mean()
            my_cnot = iter_res[:, 2].mean()
            my_dep = iter_res[:, 3].mean()
            qiskit_acc = iter_res[:, 4].mean()
            qiskit_qubits = iter_res[:, 5].mean()
            qiskit_cnot = iter_res[:, 6].mean()
            qiskit_dep = iter_res[:, 7].mean()
            temp_res = (
                num_assets,
                epsilon,
                my_acc,
                my_qubits,
                my_cnot,
                my_dep,
                qiskit_acc,
                qiskit_qubits,
                qiskit_cnot,
                qiskit_dep,
            )
            print(temp_res)
            result.append(temp_res)
    column_names = [
        "num_assets",
        "epsilon",
        "my_acc",
        "my_qubits",
        "my_cnot",
        "my_dep",
        "qiskit_acc",
        "qiskit_qubits",
        "qiskit_cnot",
        "qiskit_dep",
    ]
    df = pd.DataFrame(result, columns=column_names)
    df.to_csv(f"result/main-new.csv", index=False)


def get_complexity_reduction(_epsilon):
    epsilon = 1 / _epsilon
    results = []
    s = (0.19561288, 0.00044902, -0.00167655)
    year = 2022
    stock_file = f"./data/Nasdaq-100-{year}.xlsx"
    # read excel take mach time
    stock_data = pd.read_excel(stock_file, header=(0, 1), index_col=0)
    for num_assets in range(2, 7, 1):
        print(f"num_assets = {num_assets}")
        for _ in range(10):
            tickers = get_tickers(num_assets=num_assets, year=year)
            # print(f"tickers = {tickers}")
            value_matrix = np.array(
                [stock_data[c]["Adj Close"] for c in tickers]
            )
            period_return = value_matrix[:, 1:] / value_matrix[:, :-1] - 1
            R, S, P = (
                np.mean(period_return, axis=1),  # R
                np.cov(period_return, ddof=1),  # \Sigma
                np.ones(num_assets),  # \Pi
            )
            original_matrix = PortfolioOptimizer(R, S, P, 1, 1, 1).matrix
            or_eigenvalues = np.linalg.eigvals(original_matrix)
            or_lambda_min = np.min(np.abs(or_eigenvalues))
            or_lambda_max = np.max(np.abs(or_eigenvalues))

            optimized_matrix = PortfolioOptimizer(R, S, P, *s).matrix
            op_eigenvalues = np.linalg.eigvals(optimized_matrix)
            op_lambda_min = np.min(np.abs(op_eigenvalues))
            op_lambda_max = np.max(np.abs(op_eigenvalues))

            or_condition_number = or_lambda_max / or_lambda_min
            op_condition_number = op_lambda_max / op_lambda_min
            or_phase_qubit_num = int(
                2 + np.ceil(np.log2(or_condition_number / epsilon))
            )
            op_phase_qubit_num = int(
                2 + np.ceil(np.log2(op_condition_number / epsilon))
            )
            vector_qubit_num = int(np.ceil(np.log2(2 + num_assets)))

            or_num_qubit = or_phase_qubit_num + vector_qubit_num + 1
            op_num_qubit = op_phase_qubit_num + vector_qubit_num + 1
            or_depth = calculate_resource(
                or_phase_qubit_num, vector_qubit_num, "Depth"
            )
            op_depth = calculate_resource(
                op_phase_qubit_num, vector_qubit_num, "Depth"
            )
            or_gate_count = calculate_resource(
                or_phase_qubit_num, vector_qubit_num, "CNOT"
            )
            op_gate_count = calculate_resource(
                op_phase_qubit_num, vector_qubit_num, "CNOT"
            )

            temp_result = (
                num_assets,
                tickers,
                op_depth,
                op_gate_count,
                op_num_qubit,
                or_depth,
                or_gate_count,
                or_num_qubit,
            )
            results.append(temp_result)

    column_names = [
        "num_tickers",
        "tickers",
        "optimized_depth",
        "optimized_gate_count",
        "optimized_num_qubit",
        "original_depth",
        "original_gate_count",
        "original_num_qubit",
    ]
    df = pd.DataFrame(results, columns=column_names)
    df.to_csv(f"result/s/complexity_reduction_{_epsilon}.csv", index=False)


def get_accuracy_improvement(nb):
    results = []
    s = (0.19561288, 0.00044902, -0.00167655)
    year = 2022
    stock_file = f"./data/Nasdaq-100-{year}.xlsx"
    # read excel take mach time
    stock_data = pd.read_excel(stock_file, header=(0, 1), index_col=0)
    for num_assets in range(2, 7, 1):
        print("num_assets", num_assets)
        for iter in range(10):
            print(f"第{iter}次")
            tickers = get_tickers(num_assets=num_assets, year=year)
            # print(f"tickers = {tickers}")
            value_matrix = np.array(
                [stock_data[c]["Adj Close"] for c in tickers]
            )
            period_return = value_matrix[:, 1:] / value_matrix[:, :-1] - 1
            R, S, P = (
                np.mean(period_return, axis=1),  # R
                np.cov(period_return, ddof=1),  # \Sigma
                np.ones(num_assets),  # \Pi
            )
            or_matrix, or_b = PortfolioOptimizer(R, S, P, 1, 1, 1).equation
            or_eigenvalues = np.linalg.eigvals(or_matrix)
            or_lambda_min = np.min(np.abs(or_eigenvalues))
            or_lambda_max = np.max(np.abs(or_eigenvalues))

            op_matrix, op_b = PortfolioOptimizer(R, S, P, *s).equation
            op_eigenvalues = np.linalg.eigvals(op_matrix)
            op_lambda_min = np.min(np.abs(op_eigenvalues))
            op_lambda_max = np.max(np.abs(op_eigenvalues))

            or_result = HHLSolver(
                or_matrix,
                or_b,
                or_lambda_min,
                or_lambda_max,
                phase_qubit_num=nb,
            ).solve()[2: 2 + num_assets]

            op_result = HHLSolver(
                op_matrix,
                op_b,
                op_lambda_min,
                op_lambda_max,
                phase_qubit_num=nb,
            ).solve()[2: 2 + num_assets]

            std_result = NumpySolver(or_matrix, or_b).solve()[
                2: 2 + num_assets
            ]
            or_acc = cal_cos_similarity(std_result, or_result)
            op_acc = cal_cos_similarity(std_result, op_result)
            temp_result = (num_assets, tickers, or_acc, op_acc)
            results.append(temp_result)

    column_names = ["num_tickers", "tickers", "or_acc", "op_acc"]
    df = pd.DataFrame(results, columns=column_names)
    df.to_csv(f"result/s/accuracy_improvement_{nb}.csv", index=False)


if __name__ == "__main__":
    set_random()
    if sys.argv[1] == "1":
        qiskit_acc_fixed_qubit(int(sys.argv[2]))
    elif sys.argv[1] == "2":
        qiskit_acc_fixed_stock(
            int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
        )
    elif sys.argv[1] == "3":
        hybrid_acc_fixed_qubit(int(sys.argv[2]))
    elif sys.argv[1] == "4":
        hybrid_acc_fixed_stock(
            int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
        )
    elif sys.argv[1] == "5":
        sapo_without_scale_fix_qubit(int(sys.argv[2]))
    elif sys.argv[1] == "6":
        sapo_without_scale_fix_stock(
            int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
        )
    elif sys.argv[1] == "7":
        sapo_without_eigen_fix_qubit(int(sys.argv[2]))
    elif sys.argv[1] == "8":
        sapo_without_eigen_fix_stock(
            int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
        )
    elif sys.argv[1] == "9":
        sapo_fix_qubit(int(sys.argv[2]))
    elif sys.argv[1] == "10":
        sapo_fix_stock(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
