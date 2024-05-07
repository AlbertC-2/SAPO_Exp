import random
import numpy as np
from typing import List, Union, Optional, Tuple


def cal_cos_similarity(x, y):
    """
    计算两个向量的余弦相似度

    参数:
        x: 第一个向量
        y: 第二个向量

    返回值:
        两个向量的余弦相似度

    """
    return np.dot(x, y) / np.linalg.norm(x) / np.linalg.norm(y)


def calculate_resource(
    phase_qubit_num,  # phase_qubit_num
    vector_qubit_num,
    an_type,  # 'Depth' or 'CNOT'
):
    params_dict = {
        (2, "Depth"): {"a": 132, "b": 2, "c": 0, "d": 4, "e": -133},
        (2, "CNOT"): {"a": 69, "b": 2, "c": 2, "d": -2, "e": -67},
        (3, "Depth"): {"a": 792, "b": 2, "c": 0, "d": 4, "e": -789},
        (3, "CNOT"): {"a": 417, "b": 2, "c": 2, "d": -2, "e": -413},
        (4, "Depth"): {"a": 3856, "b": 2, "c": 0, "d": 4, "e": -3845},
        (4, "CNOT"): {"a": 2033, "b": 2, "c": 2, "d": -2, "e": -2025},
    }
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

    an = (
        a * b**phase_qubit_num
        + c * phase_qubit_num**2
        + d * phase_qubit_num
        + e
    )
    return an


class PortfolioOptimizer:
    def __init__(
        self,
        income_mean: Union[np.ndarray, List],  # R
        income_cov: Union[np.ndarray, List],  # Sigma
        price: Optional[Union[np.array, List]],  # Pi
        s1: float = 1,
        s2: float = 1,
        s3: float = 1,  # 在某些情况下不需要, i.e. num = 2
        budget: float = 1,  # 预算 (default 1)
        expected_income: float = 1,  # 预期收入
        solver: Optional[str] = "hhl",
    ) -> None:
        if np.ndim(income_mean) != 1:
            raise ValueError(
                "Input `income_mean` must be a one-dimensional vector!"
            )
        if np.ndim(income_cov) != 2:
            raise ValueError(
                "Input `income_cov` must be a two-dimensional matrix!"
            )

        self.num_assets = np.shape(income_mean)[0]
        if isinstance(income_mean, list):
            income_mean = np.array(income_mean)
        if isinstance(income_cov, list):
            income_cov = np.array(income_cov)

        if price is None:
            price = np.ones(self.num_assets)

        self.income_mean = income_mean
        self.income_cov = income_cov
        self.price = price
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3
        self.budget = budget
        self.expected_income = expected_income

        self._matrix = None
        self._vector = None
        self._solver = solver

    def _construct_linear_equation(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._construct_matrix(), self._construct_vector()

    def _construct_matrix(self) -> np.ndarray:
        line1 = np.concatenate(([0] * 2, self.income_mean * self.s1), axis=0)
        line2 = np.concatenate(([0] * 2, self.price * self.s2), axis=0)
        line3 = np.concatenate(
            (
                np.reshape(self.income_mean * self.s1, (-1, 1)),
                np.reshape(self.price * self.s2, (-1, 1)),
                self.income_cov,
            ),
            axis=1,
        )

        A = np.concatenate(
            (np.reshape(line1, (1, -1)), np.reshape(line2, (1, -1)), line3),
            axis=0,
        )
        # print(f"A = {A}")

        if self._solver == "numpy":
            return A
        elif self._solver == "hhl":
            initial_size = np.shape(A)[0]
            modified_size = int(2 ** np.ceil(np.log2(initial_size)))
            Ca = np.r_[
                np.c_[
                    A, np.zeros((initial_size, modified_size - initial_size))
                ],
                np.c_[
                    np.zeros((modified_size - initial_size, initial_size)),
                    np.identity(modified_size - initial_size) * self.s3,
                ],
            ]

            if not np.all(Ca == Ca.T):
                raise ValueError("寄")
            else:
                return Ca
            # if np.all(Ca == Ca.T):
            #     print("WWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWWW")
            # return np.r_[
            #     np.c_[np.zeros_like(Ca), Ca],
            #     np.c_[np.transpose(Ca), np.zeros_like(Ca)],
            # ]
        else:
            raise ValueError("No such method!")

    def _construct_vector(self) -> np.ndarray:
        b = np.concatenate(
            (
                [self.expected_income * self.s1, self.budget * self.s2],
                np.zeros(self.num_assets),
            ),
            axis=0,
        )
        if self._solver == "numpy":
            return b
        elif self._solver == "hhl":
            initial_size = np.shape(b)[0]
            modified_size = int(2 ** np.ceil(np.log2(initial_size)))
            Cb = np.r_[b, np.zeros(modified_size - initial_size)]
            return Cb

    def _print_info(self, matrix):
        print(f"asset num: {self.num_assets}.")
        print(f"matrix size: {np.shape(matrix)}.")
        print(f"kappa: {np.linalg.cond(matrix)}.")
        print(f"lambda: {np.linalg.eigvals(matrix)}.")

    @property
    def matrix(self) -> np.ndarray:
        if self._matrix is None:
            self._matrix = self._construct_matrix()
        return self._matrix

    @property
    def vector(self) -> np.ndarray:
        if self._vector is None:
            self._vector = self._construct_vector()
        return self._vector

    @property
    def equation(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.matrix, self.vector


def set_random(
    seed_value=42,
):
    np.random.seed(seed_value)
    random.seed(seed_value)


def construct_data(num_assets, R, S, matrix_scale, s):
    length = int(num_assets + (num_assets**2 + num_assets) // 2)
    X = np.empty((0, length))
    R_array = R * matrix_scale * s[0]
    upper_S = S[np.triu_indices(S.shape[0])]
    sample = np.concatenate((R_array, upper_S))
    X = np.vstack((X, sample))
    return X
