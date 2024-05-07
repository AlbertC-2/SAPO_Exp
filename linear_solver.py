from abc import ABC, abstractmethod
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from matplotlib import pyplot as plt
import numpy as np
import scipy as sp
from qiskit.quantum_info import Statevector
from typing import List, Tuple, Union
from qiskit.circuit.library import PhaseEstimation, ExactReciprocal
from qiskit.circuit import AncillaRegister
from qiskit.opflow import I, Z, StateFn, TensoredOp
from qiskit.utils import algorithm_globals
import pickle

algorithm_globals.massive = True


class AbstractSolver(ABC):
    def __init__(
        self, matrix: Union[List, np.ndarray], vector: Union[List, np.ndarray]
    ) -> None:
        self.matrix = matrix
        self.vector = vector

    def _print_complexity(self) -> None:
        print(f"matrix size: {np.shape(self.matrix)}.")
        print(f"kappa: {np.linalg.cond(self.matrix)}.")
        print(f"lambda: {np.linalg.eigvals(self.matrix)}")

    @abstractmethod
    def solve(self) -> np.ndarray:
        self._print_complexity()


class NumpySolver(AbstractSolver):
    def __init__(
        self, matrix: Union[List, np.ndarray], vector: Union[List, np.ndarray]
    ) -> None:
        super().__init__(matrix, vector)

    def solve(self) -> np.ndarray:
        return np.linalg.solve(self.matrix, self.vector)


class QiskitSolver(AbstractSolver):
    def __init__(self, matrix, vector, phase_qubit_num: int) -> None:
        self.vector = vector
        self.matrix = matrix
        self._scaling = None
        self.scaling = 1
        self.phase_qubit_num = phase_qubit_num

    @property
    def scaling(self) -> float:
        """The scaling of the solution vector."""
        return self._scaling

    @scaling.setter
    def scaling(self, scaling: float) -> None:
        """Set the new scaling of the solution vector."""
        self._scaling = scaling

    def _calculate_norm(self, qc: QuantumCircuit) -> float:
        """Calculates the value of the euclidean norm of the solution.

        Args:
                qc: The quantum circuit preparing the solution x to the system.

        Returns:
                The value of the euclidean norm of the solution.
        """
        # Calculate the number of qubits
        nb = qc.qregs[0].size
        nl = qc.qregs[1].size
        na = qc.num_ancillas

        # Create the Operators Zero and One
        zero_op = (I + Z) / 2
        one_op = (I - Z) / 2

        # Norm observable
        observable = one_op ^ TensoredOp((nl + na) * [zero_op]) ^ (I ^ nb)
        norm_2 = (~StateFn(observable) @ StateFn(qc)).eval()

        return np.real(np.sqrt(norm_2) / self.scaling)

    def construct_circuit(self) -> QuantumCircuit:
        vector = self.vector
        nb = int(np.log2(len(vector)))
        vector_circuit = QuantumCircuit(nb)
        vector_circuit.iso(
            vector / np.linalg.norm(vector), list(range(nb)), None
        )
        matrix_circuit = QuantumCircuit(nb)
        matrix_circuit.unitary(
            sp.linalg.expm(2j * self.matrix * np.pi),
            matrix_circuit.qubits,
        )
        kappa = 1
        nl = max(nb + 1, int(np.ceil(np.log2(kappa + 1)))) + 1
        # NOTE: 这是手动设置的，为了获取固定比特数不同stock的精度
        nl = self.phase_qubit_num
        nf = 1
        delta = 1 / (2**nl)
        reciprocal_circuit = ExactReciprocal(nl, delta, neg_vals=True)
        qb = QuantumRegister(nb)  # right hand side and solution
        ql = QuantumRegister(nl)  # eigenvalue evaluation qubits
        qf = QuantumRegister(nf)  # flag qubits
        qc = QuantumCircuit(qb, ql, qf)
        # State preparation
        qc.append(vector_circuit, qb[:])
        # QPE
        phase_estimation = PhaseEstimation(nl, matrix_circuit)
        qc.append(phase_estimation, ql[:] + qb[:])
        qc.append(reciprocal_circuit, ql[::-1] + [qf[0]])
        qc.append(phase_estimation.inverse(), ql[:] + qb[:])
        return qc

    def solve(self, circuit=None):
        if circuit is None:
            circuit = self.construct_circuit()
        print("构造电路完成")
        norm = self._calculate_norm(circuit)
        print("计算norm完成")
        state = np.real(Statevector(circuit).data)[
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
        print("获取state完成")
        return norm * state / np.linalg.norm(state)


class HybridSolver(AbstractSolver):
    def __init__(
        self,
        matrix,  # 缩放过的矩阵
        vector,  # 缩放过的向量
        lambda_min,  # 将SVM计算得到的特征值结果直接传给这个类
        lambda_max,  # 后面可以考虑在外面包一个类, 加上 SVM
        *,
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
        abs_lambda_min = max(
            abs_lambda_min, 1 / (2 ** (self.phase_qubit_num + 1))
        )
        abs_lambda_max = self.lambda_max * (1 + self.disturbance[1])
        # condition_number = abs_lambda_max / abs_lambda_min
        trace = np.trace(self.matrix)
        self.scaling /= trace
        self.matrix /= trace

        if self.phase_qubit_num is None:
            print("It is impossible.")

        self.norm_const = abs_lambda_min

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
        reg_s = QuantumRegister(
            int(np.log2(np.shape(self.vector))), name="vector"
        )
        reg_r = QuantumRegister(self.phase_qubit_num, "phases")
        reg_a = QuantumRegister(1, name="flag")

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

    def solve(self, circuit=None):
        if circuit is None:
            circuit = self.construct_circuit()
        print("构造电路完成")
        ss = Statevector(circuit)
        print("获取state完成")
        statevector = ss.data
        statevector_real = np.real(statevector)
        # 计算最高位为1的态前面系数的平方和
        probability = ss.probabilities_dict()
        success_probability = 0
        for key, value in probability.items():
            if key[0] == "1":
                success_probability += value
        print("successsss", success_probability)
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


class HHLSolver(AbstractSolver):
    def __init__(
        self,
        matrix,  # 缩放过的矩阵
        vector,  # 缩放过的向量
        lambda_min,  # 将SVM计算得到的特征值结果直接传给这个类
        lambda_max,  # 后面可以考虑在外面包一个类, 加上 SVM
        *,
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

        self.norm_const = abs_lambda_min

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
        reg_s = QuantumRegister(
            int(np.log2(np.shape(self.vector))), name="vector"
        )
        reg_r = QuantumRegister(self.phase_qubit_num, "phases")
        reg_a = QuantumRegister(1, name="flag")

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

    def solve(self, circuit=None):
        if circuit is None:
            circuit = self.construct_circuit()
        print("构造电路完成")
        ss = Statevector(circuit)
        print("获取state完成")
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

        print("构造电路完成")
        norm = self._calculate_norm(circuit)
        print("计算norm完成")
        state = np.real(Statevector(circuit).data)[
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
        print("获取state完成")
        return self.scaling * state * norm / np.linalg.norm(state)

    def _calculate_norm(self, qc: QuantumCircuit) -> float:
        nb = qc.qregs[0].size
        nl = qc.qregs[1].size
        zero_op = (I + Z) / 2
        one_op = (I - Z) / 2
        observable = one_op ^ TensoredOp(nl * [zero_op]) ^ (I ^ nb)
        success_probability = (~StateFn(observable) @ StateFn(qc)).eval()
        return np.real(np.sqrt(success_probability) / self.norm_const)


def solve_linear_system(A, b, draw_circuit=False):
    """未完待续"""

    def get_lambda(A):
        pass

    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    lambda_min, lambda_max = get_lambda(A)
    print(A)
    if draw_circuit:
        HHLSolver(A, b, phase_qubit_num=4).construct_circuit(True).decompose(
            "QPE", reps=2
        ).draw("mpl")
        plt.show()
    classic_result = NumpySolver(A, b).solve()
    hhl_result = HHLSolver(A, b, lambda_min, lambda_max).solve()
    print(f"classic result: {classic_result}")
    print(f"hhl result: {hhl_result}")
    print(
        f"精度：{np.dot(hhl_result,classic_result)/np.linalg.norm(classic_result)/np.linalg.norm(hhl_result)}"
    )


def test_example():
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

    from data_provider import get_tickers
    from data_provider import generate_portfolio_problem_raw
    from util import PortfolioOptimizer

    np.set_printoptions(linewidth=np.inf)

    matrix_scale = 4000
    s = (0.19561288, 0.00044902, -0.00167655)
    num_assets = 2
    year = 2022
    tickers = get_tickers(num_assets=num_assets, year=year)
    stock_file = f"./data/Nasdaq-100-{year}.xlsx"
    R, S, P = generate_portfolio_problem_raw(
        tickers=tickers, stock_file=stock_file
    )
    optimized_matrix, b = PortfolioOptimizer(R, S, P, *s).equation
    for type in ["min", "max"]:
        model_filename = (
            f"result/svm/s/svm_model_{type}_{num_assets}_tickers_ss.pkl"
        )

        with open(model_filename, "rb") as file:
            svm_model = pickle.load(file)
        X = construct_data(R, S)
        lambda_m = svm_model.predict(X)
        # lambda_m = 1.15593399
        # print(lambda_m)  # [1.15593399]
        if type == "min":
            # np.abs: SVR 的输出可能为负数, 可以考虑增加 matrix_scale
            # 使得 label 的值远离0点
            lambda_min = np.abs(lambda_m[0] / matrix_scale)
        elif type == "max":
            lambda_max = np.abs(lambda_m[0] / matrix_scale)

    eigenvalues = np.linalg.eigvals(optimized_matrix)
    np_lambda_min = np.min(np.abs(eigenvalues))
    np_lambda_max = np.max(np.abs(eigenvalues))
    print(f"SVM lambda = {lambda_min}, {lambda_max}")
    print(f"np  lambda = {np_lambda_min}, {np_lambda_max}")
    print(f"SVM cond = {lambda_max / lambda_min}")
    print(f"np  cond = {np_lambda_max / np_lambda_min}")

    A = np.array(optimized_matrix, dtype=float)
    classic_result = NumpySolver(A, b).solve()[2 : 2 + num_assets]
    hhl_result = HHLSolver(A, b, lambda_min, lambda_max).solve()[
        2 : 2 + num_assets
    ]
    qiskit_result = QiskitSolver(A, b, phase_qubit_num=4).solve()[
        2 : 2 + num_assets
    ]
    print(f"classic result: {classic_result}")
    print(f"hhl result: {hhl_result}")
    print(
        f"余弦相似度：{np.dot(hhl_result,classic_result)/np.linalg.norm(classic_result)/np.linalg.norm(hhl_result)}"
    )
    print(
        f"余弦相似度：{np.dot(qiskit_result,classic_result)/np.linalg.norm(classic_result)/np.linalg.norm(qiskit_result)}"
    )


if __name__ == "__main__":
    test_example()
