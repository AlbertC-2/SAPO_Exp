
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys


def read_data():
    path = 'result/main/fix-stock/kappa_v3.csv'
    data = pd.read_csv(path)
    x = data[['num_qubits', 'kappa_opt', 'kappa_original']]
    list_x = x.values
    return list_x


def process_data_kappa_midian(list_x):
    num_assets = []
    kappa_opt_median = []
    kappa_origin_median = []
    kappa_opt_t = []
    kappa_origin_t = []
    count = 0

    last_num = 0
    for row in list_x:
        kappa_opt_t = []
        kappa_origin_t = []
        if last_num != row[0] and count != 0:
            num_assets.append(row[0])
            kappa_opt_median.append(np.median(kappa_opt_t))
            kappa_origin_median.append(np.median(kappa_origin_t))
            count = 0

        last_num = row[0]
        kappa_opt_t.append(row[1])
        kappa_origin_t.append(row[2])
        count += 1
    num_assets.append(row[0])
    kappa_opt_median.append(np.median(kappa_opt_t))
    kappa_origin_median.append(np.median(kappa_origin_t))

    return num_assets, kappa_opt_median, kappa_origin_median


def process_data(list_x):
    num_assets = []
    kappa_opt_mean = []
    kappa_origin_mean = []
    count = 0
    sum_opt = 0
    sum_orig = 0
    last_num = 0
    for row in list_x:
        if last_num != row[0] and count != 0:
            num_assets.append(row[0])
            kappa_opt_mean.append(sum_opt/count)
            kappa_origin_mean.append(sum_orig/count)
            count = 0
            sum_orig = 0
            sum_opt = 0
        last_num = row[0]
        sum_opt += row[1]
        sum_orig += row[2]
        count += 1
    num_assets.append(row[0])
    kappa_opt_mean.append(sum_opt/count)
    kappa_origin_mean.append(sum_orig/count)
    return num_assets, kappa_opt_mean, kappa_origin_mean


def plot_kappa(num_assets, kappa_opt_mean, kappa_origin_mean):
    # print(kappa_opt_mean)
    # print(kappa_origin_mean)
    plt.plot(num_assets, kappa_opt_mean, 'b*-', alpha=0.5,
             linewidth=1, label='$\kappa$ of SAPO')  # 'bo-'表示蓝色实线，数据点实心原点标注
    plt.plot(num_assets, kappa_origin_mean, 'r*-', alpha=0.5,
             linewidth=1, label='$\kappa$ without optimization')  # 'bo-'表示蓝色实线，数据点实心原点标注
    # plot中参数的含义分别是横轴值，纵轴值，线的形状（'s'方块,'o'实心圆点，'*'五角星   ...，颜色，透明度,线的宽度和标签 ，

    polynomial_opt = np.polyfit(num_assets, kappa_opt_mean, 5)
    polynomial_orig = np.polyfit(num_assets, kappa_origin_mean, 5)
    print("polynomial_opt polynomial coefficients:", polynomial_opt)
    print("polynomial_orig polynomial coefficients:", polynomial_orig)
    p_opt = np.poly1d(polynomial_opt)
    p_orig = np.poly1d(polynomial_orig)

    y_orig = p_orig(num_assets)  # 拟合结果
    plt.plot(num_assets, y_orig, 'g*--', alpha=0.5,
             linewidth=1, label='$\kappa_{orig}$ function fitting')  # 'bo-'表示蓝色实线，数据点实心原点标注

    plt.legend()  # 显示上面的label
    plt.xlabel('number of assets')  # x_label
    plt.ylabel('$\kappa$ (condition number)')  # y_label
    plt.show()


def polynomial_fitting(num_assets, kappa_opt_mean, kappa_origin_mean):
    polynomial_opt = np.polyfit(num_assets, kappa_opt_mean, 5)
    polynomial_orig = np.polyfit(num_assets, kappa_origin_mean, 5)
    print("polynomial_opt polynomial coefficients:", polynomial_opt)
    print("polynomial_orig polynomial coefficients:", polynomial_orig)
    p_opt = np.poly1d(polynomial_opt)
    p_orig = np.poly1d(polynomial_orig)
    return p_opt, p_orig


def plot_complexity1(num_assets, kappa_opt_mean, kappa_origin_mean, p_opt, p_orig):
    epsilon = 1/8
    time_gauss = []
    time_gc = []
    time_hhl = []
    time_sapo = []

    for i in range(len(num_assets)):
        time_hhl.append(np.log(int(2 ** np.ceil(np.log2(num_assets[i]+2)))) *
                        kappa_origin_mean[i]**2*0.25*1/epsilon)
        time_sapo.append(np.log(int(2 ** np.ceil(np.log2(num_assets[i]+2))))
                         * kappa_opt_mean[i]**2*0.25*1/epsilon)

    plt.plot(num_assets, time_hhl, 'g*-', alpha=0.5,
             linewidth=1, label='time of HHL')  # 'bo-'表示蓝色实线，数据点实心原点标注
    plt.plot(num_assets, time_sapo, 'r*-', alpha=0.5,
             linewidth=1, label='time of SAPO')  # 'bo-'表示蓝色实线，数据点实心原点标注
    plt.legend()  # 显示上面的label
    plt.xlabel('number of assets')  # x_label
    plt.ylabel('time')  # y_label
    plt.show()


def plot_complexity4(num_assets, kappa_opt_mean, kappa_origin_mean, p_opt, p_orig):
    """
    use polynomial fitting method to deal with larger number of assets
    """
    epsilon = 1/8
    time_hhl = []
    time_sapo = []

    for num in num_assets:
        time_hhl.append(np.log(
            int(2 ** np.ceil(np.log2(num+2))))*p_orig(num)**2*0.25*1/epsilon)
        time_sapo.append(np.log(
            int(2 ** np.ceil(np.log2(num+2))))*p_opt(num)**2*0.25*1/epsilon)
    time_hhl = np.log(time_hhl)
    time_sapo = np.log(time_sapo)
    plt.plot(num_assets, time_hhl, 'g*-', alpha=0.5,
             linewidth=1, label='logarithmic time of HHL')  # 'bo-'表示蓝色实线，数据点实心原点标注
    plt.plot(num_assets, time_sapo, 'r*-', alpha=0.5,
             linewidth=1, label='logarithmic time of SAPO')  # 'bo-'表示蓝色实线，数据点实心原点标注
    plt.legend()  # 显示上面的label
    plt.xlabel('number of assets')  # x_label
    plt.ylabel('logarithmic time')  # y_label
    plt.show()


def plot_complexity2(num_assets, kappa_opt_mean, kappa_origin_mean, p_opt, p_orig):
    epsilon = 1/8
    time_gauss = []
    time_gc = []
    time_hhl = []
    time_sapo = []
    for num in num_assets:
        time_gauss.append(num**3)
        # ac: actually, the kappa of gc is not the same of hhl
        time_gc.append(num*p_orig(num)*np.log(1/epsilon))
        time_hhl.append(np.log(
            int(2 ** np.ceil(np.log2(num+2))))*p_orig(num)**2*0.25*1/epsilon)
        time_sapo.append(np.log(
            int(2 ** np.ceil(np.log2(num+2))))*p_opt(num)**2*0.25*1/epsilon)

    plt.plot(num_assets, time_gauss, 'b*-', alpha=0.5,
             linewidth=1, label='time of Gauss')  # 'bo-'表示蓝色实线，数据点实心原点标注
    plt.plot(num_assets, time_gc, 'y*-', alpha=0.5,
             linewidth=1, label='time of Gradient Conjugate')  # 'bo-'表示蓝色实线，数据点实心原点标注
    plt.plot(num_assets, time_hhl, 'g*-', alpha=0.5,
             linewidth=1, label='time of HHL')  # 'bo-'表示蓝色实线，数据点实心原点标注
    plt.plot(num_assets, time_sapo, 'r*-', alpha=0.5,
             linewidth=1, label='time of SAPO')  # 'bo-'表示蓝色实线，数据点实心原点标注
    plt.legend()  # 显示上面的label
    plt.xlabel('number of assets')  # x_label
    plt.ylabel('time')  # y_label
    plt.show()


def plot_complexity5(num_assets, kappa_opt_mean, kappa_origin_mean, p_opt, p_orig):
    epsilon = 1/8
    time_gauss = []
    time_gc = []
    time_hhl = []
    time_sapo = []
    for num in num_assets:
        # ac: actually, the kappa of gc is not the same of hhl
        time_gc.append(num*p_orig(num)*np.log(1/epsilon))
        time_hhl.append(np.log(
            int(2 ** np.ceil(np.log2(num+2))))*p_orig(num)**2*0.25*1/epsilon)
        time_sapo.append(np.log(
            int(2 ** np.ceil(np.log2(num+2))))*p_opt(num)**2*0.25*1/epsilon)

    plt.plot(num_assets, time_gc, 'y*-', alpha=0.5,
             linewidth=1, label='time of Gradient Conjugate')  # 'bo-'表示蓝色实线，数据点实心原点标注
    plt.plot(num_assets, time_hhl, 'g*-', alpha=0.5,
             linewidth=1, label='time of HHL')  # 'bo-'表示蓝色实线，数据点实心原点标注
    plt.plot(num_assets, time_sapo, 'r*-', alpha=0.5,
             linewidth=1, label='time of SAPO')  # 'bo-'表示蓝色实线，数据点实心原点标注
    plt.legend()  # 显示上面的label
    plt.xlabel('number of assets')  # x_label
    plt.ylabel('time')  # y_label
    plt.show()


def plot_complexity3(num_assets, kappa_opt_mean, kappa_origin_mean, p_opt, p_orig):
    epsilon = 1/8
    time_gauss = []
    time_gc = []
    time_hhl = []
    time_sapo = []
    # not considering kappa
    nums = list(range(1000, 500000, 5000))
    for num in nums:
        time_gauss.append(num**3)
        time_gc.append(num*np.log(1/epsilon))
        # time_hhl.append(np.log(num)*0.25*1/epsilon)
        time_sapo.append(
            np.log(int(2 ** np.ceil(np.log2(num+2))))*0.25*1/epsilon)
    time_gauss = np.log(time_gauss)
    time_gc = np.log(time_gc)
    time_sapo = np.log(time_sapo)
    plt.plot(nums, time_gauss, 'b*-', alpha=0.5,
             linewidth=1, label='logarithmic time of Gauss')  # 'bo-'表示蓝色实线，数据点实心原点标注
    plt.plot(nums, time_gc, 'y*-', alpha=0.5,
             linewidth=1, label='logarithmic time of Gradient Conjugate')  # 'bo-'表示蓝色实线，数据点实心原点标注
    plt.plot(nums, time_sapo, 'r*-', alpha=0.5,
             linewidth=1, label='logarithmic time of SAPO')  # 'bo-'表示蓝色实线，数据点实心原点标注
    plt.legend()  # 显示上面的label
    plt.xlabel('number of assets')  # x_label
    plt.ylabel('logarithmic time')  # y_label
    plt.show()


if __name__ == "__main__":
    data = read_data()
    num_assets, kappa_opt_mean, kappa_origin_mean = process_data(data)
    p_opt, p_orig = polynomial_fitting(
        num_assets, kappa_opt_mean, kappa_origin_mean)
    num_assets_large = [1000, 5000, 10000, 30000,
                        60000, 100000, 200000, 400000, 600000]
    if sys.argv[1] == "1":
        plot_kappa(num_assets, kappa_opt_mean, kappa_origin_mean)
    elif sys.argv[1] == "2":
        plot_complexity1(num_assets, kappa_opt_mean,
                         kappa_origin_mean, p_opt, p_orig)
    elif sys.argv[1] == "3":
        plot_complexity2(num_assets, kappa_opt_mean,
                         kappa_origin_mean, p_opt, p_orig)
    elif sys.argv[1] == "4":
        plot_complexity3(num_assets, kappa_opt_mean,
                         kappa_origin_mean, p_opt, p_orig)
    elif sys.argv[1] == "5":
        plot_complexity4(num_assets_large, kappa_opt_mean,
                         kappa_origin_mean, p_opt, p_orig)
    elif sys.argv[1] == "6":
        plot_complexity5(num_assets, kappa_opt_mean,
                         kappa_origin_mean, p_opt, p_orig)
