import numpy as np
from typing import List, Union, Optional, Tuple


class PortfolioOptimizer:
    """
    利用 R, Sigma, Pi 以及缩放系数 构造矩阵
    """
    def __init__(
        self,
        income_mean: Union[np.ndarray, List],  # R
        income_cov: Union[np.ndarray, List],  # Sigma
        price: Optional[Union[np.array, List]],  # Pi
        s1: float,
        s2: float,
        s3: float = 1,
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
            ([self.expected_income, self.budget], np.zeros(self.num_assets)),
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
