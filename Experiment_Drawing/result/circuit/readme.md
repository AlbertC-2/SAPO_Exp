# 利用数列通项公式直接计算电路深度和CNOT门数量

- 跑 construct_circuit.py 获得数据
    - code-construct_circuit.py 为代码备份
- c_result_temp_*.csv 为小规模下的电路深度和CNOT门数量数据
- 在 a_circuit.ipynb 中找数列规律，并得到通项公式
- circuit_result_*.csv 为测试结果
    - 跑 linear_solver_circuit.py 可得其结果
        - code-linear_solver_circuit.py 为代码备份
    - circuit_result_10000.csv 的重复次数为 10000
- draw.ipynb 是可视化相关的代码