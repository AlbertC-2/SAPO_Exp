"""
对于不同的 num_assets, 对于用了 s 还是没有用 s, 电路的深度和CNOT门的数量如何变化
- 不同的 num_assets -> 不同的 vector_qubit_num
- 用了 s -> cond 降低 -> 在相同的精度下 phase_qubit_num 降低

然后通过 vector_qubit_num 和 phase_qubit_num 利用数组通项公式计算所需的数据
"""
import random
import pandas as pd
from abc import ABC, abstractmethod
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import numpy as np
import scipy as sp
from qiskit.quantum_info import Statevector
from typing import List, Tuple, Union
from qiskit.circuit.library import PhaseEstimation
from qiskit.opflow import I, Z, StateFn, TensoredOp
from data_provider import get_tickers
from util import PortfolioOptimizer
import math
import warnings

warnings.filterwarnings("ignore")


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


class NumpySolver(AbstractSolver):
    def __init__(
        self, matrix: Union[List, np.ndarray], vector: Union[List, np.ndarray]
    ) -> None:
        super().__init__(matrix, vector)

    def solve(self) -> np.ndarray:
        super().solve()
        return np.linalg.solve(self.matrix, self.vector)


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
        epsilon: float = 1 / 32,
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
        # start_time = time.time()
        phase_estimation = PhaseEstimation(reg_r.size, matrix_circuit)
        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # print(f"Done! construct_circuit here in {elapsed_time} sec.")
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

        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # print(f"Done! construct_circuit in {elapsed_time} sec.")
        return circuit

    def solve(self):
        super().solve()
        circuit = self.construct_circuit()
        ss = Statevector(circuit)
        statevector = ss.data
        statevector_real = np.real(statevector)
        # 计算最高位为1的态前面系数的平方和
        probability = ss.probabilities_dict()
        success_probability = 0
        for key, value in probability.items():
            if key[0] == "1":
                success_probability += value
        # print(success_probability)
        norm = np.real(np.sqrt(success_probability) / self.norm_const)

        state = statevector_real[
            [
                int(
                    "1"
                    + circuit.qregs[1].size * "0"
                    + np.binary_repr(i, width=circuit.qregs[0].size),
                    2,
                )
                for i in range(2 ** circuit.qregs[0].size)
            ]
        ]

        return self.scaling * state * norm / np.linalg.norm(state)

    def _calculate_norm(self, qc: QuantumCircuit) -> float:
        nb = qc.qregs[0].size
        nl = qc.qregs[1].size
        zero_op = (I + Z) / 2
        one_op = (I - Z) / 2
        observable = one_op ^ TensoredOp(nl * [zero_op]) ^ (I ^ nb)
        success_probability = (~StateFn(observable) @ StateFn(qc)).eval()
        print(f"In _calculate_norm: probability = {success_probability}")
        return np.real(np.sqrt(success_probability) / self.norm_const)


def calculate_an(
    n,  # phase_qubit_num
    vector_qubit_num,
    an_type,  # 'Depth' or 'CNOT'
    params_dict,
):
    params_key = (vector_qubit_num, an_type)
    params = params_dict.get(
        params_key,
        {
            "a": 0,
            "b": 0,
            "c": 0,
            "d": 0,
            "e": 0,
        },
    )

    a = params.get("a")
    b = params.get("b")
    c = params.get("c")
    d = params.get("d")
    e = params.get("e")

    an = a * b**n + c * n**2 + d * n + e
    return an


params_dict = {
    (2, "Depth"): {"a": 132, "b": 2, "c": 0, "d": 4, "e": -133},
    (2, "CNOT"): {"a": 69, "b": 2, "c": 2, "d": -2, "e": -67},
    (3, "Depth"): {"a": 792, "b": 2, "c": 0, "d": 4, "e": -789},
    (3, "CNOT"): {"a": 417, "b": 2, "c": 2, "d": -2, "e": -413},
    (4, "Depth"): {"a": 3856, "b": 2, "c": 0, "d": 4, "e": -3845},
    (4, "CNOT"): {"a": 2033, "b": 2, "c": 2, "d": -2, "e": -2025},
}


def test_circuit_exmaple():
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
    results = []

    matrix_scale = 4000
    s = (0.19561288, 0.00044902, -0.00167655)
    print(f" s = {s}")
    year = 2022
    repeat_times = 10000
    stock_file = f"./data/Nasdaq-100-{year}.xlsx"
    # read excel take mach time
    stock_data = pd.read_excel(stock_file, header=(0, 1), index_col=0)

    for num_assets in range(2, 15, 1):
        print(f"num_assets = {num_assets}")
        for i in range(repeat_times):
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
            original_matrix, or_b = PortfolioOptimizer(
                R, S, P, 1, 1, 1, budget=1, expected_income=np.random.rand()
            ).equation
            or_eigenvalues = np.linalg.eigvals(original_matrix)
            or_lambda_min = np.min(np.abs(or_eigenvalues))
            or_lambda_max = np.max(np.abs(or_eigenvalues))

            optimized_matrix, op_b = PortfolioOptimizer(
                R, S, P, *s, budget=1, expected_income=np.random.rand()
            ).equation
            op_eigenvalues = np.linalg.eigvals(optimized_matrix)
            op_lambda_min = np.min(np.abs(op_eigenvalues))
            op_lambda_max = np.max(np.abs(op_eigenvalues))

            epsilon = 1 / 32

            or_condition_number = or_lambda_max / or_lambda_min
            op_condition_number = op_lambda_max / op_lambda_min
            or_phase_qubit_num = int(
                2 + np.ceil(np.log2(or_condition_number / epsilon))
            )
            op_phase_qubit_num = int(
                2 + np.ceil(np.log2(op_condition_number / epsilon))
            )
            vector_qubit_num = math.ceil(np.log2(2 + num_assets))

            or_num_qubit = or_phase_qubit_num + vector_qubit_num + 1
            op_num_qubit = op_phase_qubit_num + vector_qubit_num + 1
            or_depth = calculate_an(
                or_phase_qubit_num, vector_qubit_num, "Depth", params_dict
            )
            op_depth = calculate_an(
                op_phase_qubit_num, vector_qubit_num, "Depth", params_dict
            )
            or_gate_count = calculate_an(
                or_phase_qubit_num, vector_qubit_num, "CNOT", params_dict
            )
            op_gate_count = calculate_an(
                op_phase_qubit_num, vector_qubit_num, "CNOT", params_dict
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
    csv_filename = "ciruit_results.csv"
    df.to_csv(csv_filename, index=False)
    print(f"save in {csv_filename}")


if __name__ == "__main__":
    tool_set_random_seed(seed_value=23)

    test_circuit_exmaple()
