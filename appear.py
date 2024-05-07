from matplotlib import pyplot as plt
import pandas as pd
import sys


def acc_fix_qubit():
    hybrid_res = pd.read_csv("./result/main/fix-qubit/hybrid-5.csv", header=0)
    qiskit_res = pd.read_csv("./result/main/fix-qubit/qiskit-5.csv", header=0)
    hybrid_res = hybrid_res.groupby("num_assets")["acc"].mean()
    qiskit_res = qiskit_res.groupby("num_assets")["acc"].mean()
    print(hybrid_res)
    print(qiskit_res)


def get_acc():
    data = pd.read_csv("result/acc/my.csv")
    data = data[["tickers", "num", "qubit_num", "similarity"]]
    average_similarity = data.groupby("num")["similarity"].mean()
    print(average_similarity)


if __name__ == "__main__":
    """
    1: 准确率
    """
    if sys.argv[1] == "1":
        acc_fix_qubit()
