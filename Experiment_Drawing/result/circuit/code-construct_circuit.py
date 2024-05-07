"""
仅构造电路, 并试图找到电路深度和门数量的规律
"""
import random
import pandas as pd
from abc import ABC, abstractmethod
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import transpile, BasicAer
import numpy as np
import scipy as sp
from typing import List, Tuple, Union
from qiskit.circuit.library import PhaseEstimation

# from data_provider import generate_hamilton_matrix
from data_provider import get_tickers
from data_provider import generate_portfolio_problem_raw
from util import PortfolioOptimizer
import time
import warnings

warnings.filterwarnings("ignore")

results = []


def tool_set_random_seed(
    seed_value=42,
):
    np.random.seed(seed_value)
    random.seed(seed_value)


class AbstractSolver(ABC):
    def __init__(
        self, matrix: Union[List, np.ndarray], vector: Union[List, np.ndarray]
    ) -> None:
        self.matrix = matrix
        self.vector = vector

    def _print_complexity(self) -> None:
        return
        # print(f"matrix size: {np.shape(self.matrix)}.")
        # print(f"kappa: {np.linalg.cond(self.matrix)}.")
        # print(f"lambda: {np.linalg.eigvals(self.matrix)}")

    @abstractmethod
    def solve(self) -> np.ndarray:
        self._print_complexity()


class HHLSolver(AbstractSolver):
    def __init__(
        self,
        matrix,  # 缩放过的矩阵
        vector,  # 缩放过的向量
        lambda_min,  # 将SVM计算得到的特征值结果直接传给这个类
        lambda_max,  # 后面可以考虑在外面包一个类, 加上 SVM
        *,
        max_qubit_num: int = 16,
        phase_qubit_num: int = None,
        epsilon: float = 0.1,  # 1/ 32
        disturbance: Tuple[float, float] = (0, 0),
    ) -> None:
        self._phase_qubit_num = None
        self.matrix = np.array(matrix)
        self.norm_const = 1
        self.scaling = np.linalg.norm(vector)
        self.vector = vector / np.linalg.norm(vector)
        self.vector_qubit_num = int(np.log2(np.shape(self.vector)))
        self.max_qubit_num = max_qubit_num
        self._phase_qubit_num = phase_qubit_num
        self.epsilon = epsilon
        self.disturbance = disturbance
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

    @property
    def phase_qubit_num(self) -> int:
        if self._phase_qubit_num is None:
            abs_lambda_min = self.lambda_min * (1 + self.disturbance[0])
            abs_lambda_max = self.lambda_max * (1 + self.disturbance[1])
            condition_number = abs_lambda_max / abs_lambda_min
            # return 2
            return int(2 + np.ceil(np.log2(condition_number / self.epsilon)))

        return self._phase_qubit_num

    def _get_parameter_values(self) -> None:
        abs_lambda_min = self.lambda_min * (1 + self.disturbance[0])
        abs_lambda_max = self.lambda_max * (1 + self.disturbance[1])
        # condition_number = abs_lambda_max / abs_lambda_min
        lambda_scaling = 0.5 / abs_lambda_max
        self.scaling *= lambda_scaling
        self.matrix *= lambda_scaling
        abs_lambda_min *= lambda_scaling
        abs_lambda_max *= lambda_scaling

        if self.phase_qubit_num is None:
            print("It is impossible.")

        self.norm_const = abs_lambda_min * 0.875

    def _construct_rotation(self, n_state_qubits: int):
        reg_state = QuantumRegister(n_state_qubits, "state")
        reg_flag = QuantumRegister(1, "flag")
        circuit = QuantumCircuit(reg_state, reg_flag, name="UCRY")
        angles = [0]
        tot = 2**n_state_qubits
        for i in range(1, tot):
            phi = i / tot
            rotation_value = (
                self.norm_const
                * 0.5
                / (phi - (i >= 2 ** (n_state_qubits - 1)))
            )
            if np.isclose(rotation_value, 1, 1e-5, 1e-5):
                angles.append(np.pi)
            elif np.isclose(rotation_value, -1):
                angles.append(-np.pi)
            elif -1 < rotation_value < 1:
                angles.append(2 * np.arcsin(rotation_value))
            else:
                angles.append(0)
        circuit.ucry(angles, reg_state[:], reg_flag[:])
        return circuit

    def construct_circuit(self, need_measurement=False):
        self._get_parameter_values()
        print(f"估计比特：{self.phase_qubit_num}")
        print(f"向量比特：{int(np.log2(np.shape(self.vector)))}")
        reg_s = QuantumRegister(
            int(np.log2(np.shape(self.vector))), name="vector"
        )
        reg_r = QuantumRegister(self.phase_qubit_num, "phases")
        reg_a = QuantumRegister(1, name="flag")
        # print("Done! construct_circuit regs.")
        vector_circuit = QuantumCircuit(reg_s.size, name="isometry")
        vector_circuit.iso(self.vector, vector_circuit.qubits, None)

        matrix_circuit = QuantumCircuit(reg_s.size, name="U")
        matrix_circuit.unitary(
            sp.linalg.expm(1j * self.matrix * np.pi),
            matrix_circuit.qubits,
        )
        phase_estimation = PhaseEstimation(reg_r.size, matrix_circuit)
        reciprocal_circuit = self._construct_rotation(reg_r.size)
        circuit = QuantumCircuit(reg_s, reg_r, reg_a)
        circuit.append(vector_circuit, reg_s[:])
        circuit.append(phase_estimation, reg_r[:] + reg_s[:])
        circuit.append(reciprocal_circuit, reg_r[::-1] + reg_a[:])
        circuit.append(phase_estimation.inverse(), reg_r[:] + reg_s[:])

        if need_measurement is True:
            reg_measurement = ClassicalRegister(1, "measure")
            circuit.add_register(reg_measurement)
            circuit.measure(reg_a[:], reg_measurement[:])

        return circuit

    def solve(self):
        super().solve()


def test_circuit():
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
        # print(f"X = {X}")
        return X

    np.set_printoptions(linewidth=np.inf)

    matrix_scale = 4000
    s = (0.19561288, 0.00044902, -0.00167655)
    print(f" s = {s}")
    # num_assets = 2
    year = 2022
    column_names = [
        "phase_qubit_num",
        "vector_qubit_num",
        "depth",
        "gate_count",
        "num_qubit",
    ]

    num_assets = 3
    for qubit_num in range(8, 16, 1):
        temp_results = []
        tickers = get_tickers(num_assets=num_assets, year=year)
        print(f"tickers = {tickers}")
        stock_file = f"./data/Nasdaq-100-{year}.xlsx"
        R, S, P = generate_portfolio_problem_raw(
            tickers=tickers, stock_file=stock_file
        )
        optimized_matrix, op_b = PortfolioOptimizer(
            R, S, P, *s, budget=1, expected_income=np.random.rand()
        ).equation
        op_A = np.array(optimized_matrix, dtype=float)
        op_eigenvalues = np.linalg.eigvals(optimized_matrix)
        op_lambda_min = np.min(np.abs(op_eigenvalues))
        op_lambda_max = np.max(np.abs(op_eigenvalues))
        # print("Start construct_circuit")
        start_time = time.time()
        optimized_circuit = HHLSolver(
            op_A,
            op_b,
            op_lambda_min,
            op_lambda_max,
            phase_qubit_num=qubit_num,
        ).construct_circuit()
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Done! construct_circuit op in {elapsed_time} sec.")

        start_time = time.time()
        op_transpiled_circuit = transpile(
            optimized_circuit, BasicAer.get_backend("qasm_simulator")
        )
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Done! transpiled_circuit op in {elapsed_time} sec.")

        op_depth = op_transpiled_circuit.depth()
        op_gate_count = op_transpiled_circuit.count_ops()["cx"]
        op_num_qubit = op_transpiled_circuit.num_qubits
        print(f"{qubit_num}, {op_depth}, {op_gate_count}, {op_num_qubit}")

        temp_result = (
            qubit_num,
            3,
            op_depth,
            op_gate_count,
            op_num_qubit,
        )
        temp_results.append(temp_result)
        results.append(temp_result)
        df = pd.DataFrame(temp_results, columns=column_names)
        df.to_csv(f"circuit/c_results_v1_{qubit_num}.csv", index=False)
    df = pd.DataFrame(results, columns=column_names)
    df.to_csv("circuit/c_results_v2.csv", index=False)


if __name__ == "__main__":
    test_circuit()
