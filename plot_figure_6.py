#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# plot_figure_6.py
# Author: Da Zhu
# E-mail: gonglisuozcd@gmail.com
# 2019-12-10 13:01:25
"""
<<地震研究>>期刊投稿图件fig6
"""
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib import rcParams
rcParams["axes.unicode_minus"] = False
rcParams["font.size"] = 14.2
rcParams["font.sans-serif"] = "STSong"
rcParams["figure.figsize"] = (10, 8)
rcParams["legend.loc"] = "upper right"
rcParams["legend.fontsize"] = 14
rcParams["legend.handlelength"] = 1.5
rcParams["legend.handletextpad"] = 0.5
rcParams["xtick.direction"] = "in"
rcParams["ytick.direction"] = "in"


def butterBand(data, lowCut, highCut, sampRat, order=4):
    f0 = 0.5 * sampRat
    low = lowCut / f0
    high = highCut / f0
    b, a = signal.butter(order, [low, high], btype='bandpass')
    butterData = signal.lfilter(b, a, data)
    return butterData


def getFFT(data, sampRat):
    dataLen = len(data)
    frc = np.linspace(0, sampRat, dataLen, endpoint=False)

    fftData = np.fft.fft(data)
    fftData = np.abs(fftData) / sampRat
    frc = frc[:int(dataLen / 2) + 1]
    fftData = fftData[:int(dataLen / 2) + 1]
    return frc, fftData


def getPower(fft_data, len_time):
    power_data = 2 * fft_data ** 2 / len_time
    power_data[0] = fft_data[0] ** 2 / len_time
    power_data[-1] = fft_data[-1] ** 2 / len_time
    return power_data


def smooth_parzen_window(power_data, len_time, band):
    df = 1 / len_time
    u = 280 / 151 / band
    udf = u * df
    lmax = int(2 / udf) + 1
    w_size = lmax * 2 - 1
    w = np.zeros(w_size)
    w[lmax - 1] = 0.75 * udf
    for i in range(1, lmax):
        dif = math.pi / 2 * i * udf
        w[lmax - 1 - i] = w[lmax - 1 + i] = w[lmax - 1] * (math.sin(dif) / dif) ** 4

    bp = lmax - 1
    conv_data = signal.convolve(power_data, w)
    for i in range(1, lmax):
        conv_data[bp + i] += conv_data[bp - i]
        conv_data[-lmax - i] += conv_data[-lmax + i]

    smooth_power = conv_data[bp:-bp]
    return smooth_power


def power2fft(power_data, len_time):
    fft_data = np.sqrt(power_data * len_time / 2)
    fft_data[0] = math.sqrt(power_data[0] * len_time)
    fft_data[-1] = math.sqrt(power_data[-1] * len_time)
    return fft_data


def calculate_fft(acc_data, sampling_freq, band):
    """计算平滑后的已滤波傅里叶谱"""
    # 进行滤波处理
    acc_data = butterBand(acc_data, 0.3, 10, sampling_freq, 4)
    data_len = len(acc_data)
    frc, fft_data = getFFT(acc_data, sampling_freq)
    power_data = getPower(fft_data, data_len / sampling_freq)
    sm_power = smooth_parzen_window(power_data, data_len / sampling_freq, band)
    sm_fft = power2fft(sm_power, data_len / sampling_freq)
    return frc, sm_fft


def plot_figure_six_a():
    file_dir = r"D:\EarthquakeResearchJournal\0-plotting\figure6\data"
    acc_data = np.loadtxt(os.path.join(file_dir, "175-LEA.txt"))
    # 数据长度
    data_length = len(acc_data[:, 0])
    # 垂直向
    v_frc_20, v_data_20 = calculate_fft(acc_data[:, 0], 200, 2.0)  # 带宽: 2.0Hz
    v_frc_10, v_data_10 = calculate_fft(acc_data[:, 0], 200, 1.0)  # 带宽: 1.0Hz
    v_frc_08, v_data_08 = calculate_fft(acc_data[:, 0], 200, 0.8)  # 带宽: 0.8Hz
    v_frc_067, v_data_067 = calculate_fft(acc_data[:, 0], 200, 0.67)  # 带宽: 0.67Hz
    v_frc_04, v_data_04 = calculate_fft(acc_data[:, 0], 200, 0.4)  # 带宽: 0.4Hz
    v_frc_02, v_data_02 = calculate_fft(acc_data[:, 0], 200, 0.2)  # 带宽: 0.2Hz
    v_frc_01, v_data_01 = calculate_fft(acc_data[:, 0], 200, 0.1)  # 带宽: 0.1Hz
    v_frc_0067, v_data_0067 = calculate_fft(acc_data[:, 0], 200, 0.067)  # 带宽: 0.067Hz
    v_frc_005, v_data_005 = calculate_fft(acc_data[:, 0], 200, 0.05)  # 带宽: 0.05Hz
    # 截断
    v_frc_20, v_data_20 = v_frc_20[:data_length // 10], v_data_20[:data_length // 10]
    v_frc_10, v_data_10 = v_frc_10[:data_length // 10], v_data_10[:data_length // 10]
    v_frc_08, v_data_08 = v_frc_08[:data_length // 10], v_data_08[:data_length // 10]
    v_frc_067, v_data_067 = v_frc_067[:data_length // 10], v_data_067[:data_length // 10]
    v_frc_04, v_data_04 = v_frc_04[:data_length // 10], v_data_04[:data_length // 10]
    v_frc_02, v_data_02 = v_frc_02[:data_length // 10], v_data_02[:data_length // 10]
    v_frc_01, v_data_01 = v_frc_01[:data_length // 10], v_data_01[:data_length // 10]
    v_frc_0067, v_data_0067 = v_frc_0067[:data_length // 10], v_data_0067[:data_length // 10]
    v_frc_005, v_data_005 = v_frc_005[:data_length // 10], v_data_005[:data_length // 10]
    # 水平合成向
    _, h1_20 = calculate_fft(acc_data[:, 1], 200, 2.0)
    _, h2_20 = calculate_fft(acc_data[:, 2], 200, 2.0)
    h_data_20 = np.sqrt(h1_20 ** 2 + h2_20 ** 2)[:data_length // 10]  # 带宽: 2.0Hz

    _, h1_10 = calculate_fft(acc_data[:, 1], 200, 1.0)
    _, h2_10 = calculate_fft(acc_data[:, 2], 200, 1.0)
    h_data_10 = np.sqrt(h1_10 ** 2 + h2_10 ** 2)[:data_length // 10]  # 带宽: 1.0Hz

    _, h1_08 = calculate_fft(acc_data[:, 1], 200, 0.8)
    _, h2_08 = calculate_fft(acc_data[:, 2], 200, 0.8)
    h_data_08 = np.sqrt(h1_08 ** 2 + h2_08 ** 2)[:data_length // 10]  # 带宽: 0.8Hz

    _, h1_067 = calculate_fft(acc_data[:, 1], 200, 0.67)
    _, h2_067 = calculate_fft(acc_data[:, 2], 200, 0.67)
    h_data_067 = np.sqrt(h1_067 ** 2 + h2_067 ** 2)[:data_length // 10]  # 带宽: 0.67Hz

    _, h1_04 = calculate_fft(acc_data[:, 1], 200, 0.4)
    _, h2_04 = calculate_fft(acc_data[:, 2], 200, 0.4)
    h_data_04 = np.sqrt(h1_04 ** 2 + h2_04 ** 2)[:data_length // 10]  # 带宽: 0.4Hz

    _, h1_02 = calculate_fft(acc_data[:, 1], 200, 0.2)
    _, h2_02 = calculate_fft(acc_data[:, 2], 200, 0.2)
    h_data_02 = np.sqrt(h1_02 ** 2 + h2_02 ** 2)[:data_length // 10]  # 带宽: 0.2Hz

    _, h1_01 = calculate_fft(acc_data[:, 1], 200, 0.1)
    _, h2_01 = calculate_fft(acc_data[:, 2], 200, 0.1)
    h_data_01 = np.sqrt(h1_01 ** 2 + h2_01 ** 2)[:data_length // 10]  # 带宽: 0.1Hz

    _, h1_0067 = calculate_fft(acc_data[:, 1], 200, 0.067)
    _, h2_0067 = calculate_fft(acc_data[:, 2], 200, 0.067)
    h_data_0067 = np.sqrt(h1_0067 ** 2 + h2_0067 ** 2)[:data_length // 10]  # 带宽: 0.067Hz

    _, h1_005 = calculate_fft(acc_data[:, 1], 200, 0.05)
    _, h2_005 = calculate_fft(acc_data[:, 2], 200, 0.05)
    h_data_005 = np.sqrt(h1_005 ** 2 + h2_005 ** 2)[:data_length // 10]  # 带宽: 0.05Hz
    # 绘图
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
    # Vertical Component
    line9, = axes[0].plot(v_frc_005, v_data_005, c="black", lw=1.0, linestyle="dotted")
    # dash dot dotted
    line1, = axes[0].plot(v_frc_20, v_data_20, c="blue", lw=1.0, linestyle="solid")
    line8, = axes[0].plot(v_frc_0067, v_data_0067, c="purple", lw=1.0, linestyle="dashed")
    # densely dashed
    line2, = axes[0].plot(v_frc_10, v_data_10, c="green", lw=1.0, linestyle=(0, (5, 1)))
    line7, = axes[0].plot(v_frc_01, v_data_01, c="gray", lw=1.0, linestyle="dashdot")
    # densely dash dotted
    line3, = axes[0].plot(v_frc_08, v_data_08, c="orange", lw=1.0, linestyle=(0, (3, 1, 1, 1)))
    # densely dash dot dotted
    line6, = axes[0].plot(v_frc_02, v_data_02, c="magenta", lw=1.0, linestyle=(0, (3, 1, 1, 1, 1, 1)))
    # loosely dash dotted
    line4, = axes[0].plot(v_frc_067, v_data_067, c="pink", lw=1.0, linestyle=(0, (3, 5, 1, 5, 1, 5)))
    line5, = axes[0].plot(v_frc_04, v_data_04, c="red", lw=2.5, linestyle="solid")
    axes[0].text(0.025, 0.8, "竖向", transform=axes[0].transAxes)
    axes[0].set_xlim([0, max(v_frc_02)])
    axes[0].set_ylim([0, 0.039])
    axes[0].set_ylabel(r"幅值/gal$\cdot$s", labelpad=1)
    axes[0].set_yticks([0, 0.01, 0.02, 0.03])
    axes[0].legend((line1, line2, line3, line4, line5, line6, line7, line8, line9),
                   ("窗带宽: 2.0 Hz",
                    "窗带宽: 1.0 Hz",
                    "窗带宽: 0.8 Hz",
                    "窗带宽: 0.67 Hz",
                    "窗带宽: 0.4 Hz",
                    "窗带宽: 0.2 Hz",
                    "窗带宽: 0.1 Hz",
                    "窗带宽: 0.067 Hz",
                    "窗带宽: 0.05 Hz"), framealpha=0.85, title="图例", ncol=3, edgecolor="k")
    # Horizontal Component
    line18, = axes[1].plot(v_frc_005, h_data_005, c="black", lw=1.0, linestyle="dotted")
    line10, = axes[1].plot(v_frc_20, h_data_20, c="blue", lw=1.0, linestyle="solid")
    line17, = axes[1].plot(v_frc_0067, h_data_0067, c="purple", lw=1.0, linestyle="dashed")
    line11, = axes[1].plot(v_frc_10, h_data_10, c="green", lw=1.0, linestyle=(0, (5, 1)))
    line16, = axes[1].plot(v_frc_01, h_data_01, c="gray", lw=1.0, linestyle="dashdot")
    line12, = axes[1].plot(v_frc_08, h_data_08, c="orange", lw=1.0, linestyle=(0, (3, 1, 1, 1)))
    line15, = axes[1].plot(v_frc_02, h_data_02, c="magenta", lw=1.0, linestyle=(0, (3, 1, 1, 1, 1, 1)))
    line13, = axes[1].plot(v_frc_067, h_data_067, c="pink", lw=1.0, linestyle=(0, (3, 5, 1, 5, 1, 5)))
    line14, = axes[1].plot(v_frc_04, h_data_04, c="red", lw=2.5, linestyle="solid")
    axes[1].text(0.025, 0.8, "水平合成向", transform=axes[1].transAxes)
    axes[1].set_xlim([0, max(v_frc_02)])
    axes[1].set_ylim([0, 0.039])
    axes[1].set_ylabel(r"幅值/gal$\cdot$s", labelpad=1)
    axes[1].set_yticks([0, 0.01, 0.02, 0.03])
    axes[1].set_xlabel("频率/Hz")
    axes[1].legend((line10, line11, line12, line13, line14, line15, line16, line17, line18),
                   ("窗带宽: 2.0 Hz",
                    "窗带宽: 1.0 Hz",
                    "窗带宽: 0.8 Hz",
                    "窗带宽: 0.67 Hz",
                    "窗带宽: 0.4 Hz",
                    "窗带宽: 0.2 Hz",
                    "窗带宽: 0.1 Hz",
                    "窗带宽: 0.067 Hz",
                    "窗带宽: 0.05 Hz"), framealpha=0.85, title="图例", ncol=3, edgecolor="k")
    plt.subplots_adjust(left=0.07, right=0.99, bottom=0.07, top=0.99, hspace=0.05, wspace=0.05)

    os.chdir(r"D:\EarthquakeResearchJournal\1-vectors")
    plt.savefig("fig6a.svg", format="svg")
    os.chdir(r"D:\EarthquakeResearchJournal\3-illustrations")
    plt.savefig("fig6a.jpg", dpi=600)
    # plt.show()


def plot_figure_six_b():
    file_dir = r"D:\EarthquakeResearchJournal\0-plotting\figure6\data"
    acc_data = np.loadtxt(os.path.join(file_dir, "181-LSA.txt"))
    # 数据长度
    data_length = len(acc_data[:, 0])
    # 垂直向
    v_frc_20, v_data_20 = calculate_fft(acc_data[:, 0], 200, 2.0)  # 带宽: 2.0Hz
    v_frc_10, v_data_10 = calculate_fft(acc_data[:, 0], 200, 1.0)  # 带宽: 1.0Hz
    v_frc_08, v_data_08 = calculate_fft(acc_data[:, 0], 200, 0.8)  # 带宽: 0.8Hz
    v_frc_067, v_data_067 = calculate_fft(acc_data[:, 0], 200, 0.67)  # 带宽: 0.67Hz
    v_frc_04, v_data_04 = calculate_fft(acc_data[:, 0], 200, 0.4)  # 带宽: 0.4Hz
    v_frc_02, v_data_02 = calculate_fft(acc_data[:, 0], 200, 0.2)  # 带宽: 0.2Hz
    v_frc_01, v_data_01 = calculate_fft(acc_data[:, 0], 200, 0.1)  # 带宽: 0.1Hz
    v_frc_0067, v_data_0067 = calculate_fft(acc_data[:, 0], 200, 0.067)  # 带宽: 0.067Hz
    v_frc_005, v_data_005 = calculate_fft(acc_data[:, 0], 200, 0.05)  # 带宽: 0.05Hz
    # 截断
    v_frc_20, v_data_20 = v_frc_20[:data_length // 10], v_data_20[:data_length // 10]
    v_frc_10, v_data_10 = v_frc_10[:data_length // 10], v_data_10[:data_length // 10]
    v_frc_08, v_data_08 = v_frc_08[:data_length // 10], v_data_08[:data_length // 10]
    v_frc_067, v_data_067 = v_frc_067[:data_length // 10], v_data_067[:data_length // 10]
    v_frc_04, v_data_04 = v_frc_04[:data_length // 10], v_data_04[:data_length // 10]
    v_frc_02, v_data_02 = v_frc_02[:data_length // 10], v_data_02[:data_length // 10]
    v_frc_01, v_data_01 = v_frc_01[:data_length // 10], v_data_01[:data_length // 10]
    v_frc_0067, v_data_0067 = v_frc_0067[:data_length // 10], v_data_0067[:data_length // 10]
    v_frc_005, v_data_005 = v_frc_005[:data_length // 10], v_data_005[:data_length // 10]
    # 水平合成向
    _, h1_20 = calculate_fft(acc_data[:, 1], 200, 2.0)
    _, h2_20 = calculate_fft(acc_data[:, 2], 200, 2.0)
    h_data_20 = np.sqrt(h1_20 ** 2 + h2_20 ** 2)[:data_length // 10]  # 带宽: 2.0Hz

    _, h1_10 = calculate_fft(acc_data[:, 1], 200, 1.0)
    _, h2_10 = calculate_fft(acc_data[:, 2], 200, 1.0)
    h_data_10 = np.sqrt(h1_10 ** 2 + h2_10 ** 2)[:data_length // 10]  # 带宽: 1.0Hz

    _, h1_08 = calculate_fft(acc_data[:, 1], 200, 0.8)
    _, h2_08 = calculate_fft(acc_data[:, 2], 200, 0.8)
    h_data_08 = np.sqrt(h1_08 ** 2 + h2_08 ** 2)[:data_length // 10]  # 带宽: 0.8Hz

    _, h1_067 = calculate_fft(acc_data[:, 1], 200, 0.67)
    _, h2_067 = calculate_fft(acc_data[:, 2], 200, 0.67)
    h_data_067 = np.sqrt(h1_067 ** 2 + h2_067 ** 2)[:data_length // 10]  # 带宽: 0.67Hz

    _, h1_04 = calculate_fft(acc_data[:, 1], 200, 0.4)
    _, h2_04 = calculate_fft(acc_data[:, 2], 200, 0.4)
    h_data_04 = np.sqrt(h1_04 ** 2 + h2_04 ** 2)[:data_length // 10]  # 带宽: 0.4Hz

    _, h1_02 = calculate_fft(acc_data[:, 1], 200, 0.2)
    _, h2_02 = calculate_fft(acc_data[:, 2], 200, 0.2)
    h_data_02 = np.sqrt(h1_02 ** 2 + h2_02 ** 2)[:data_length // 10]  # 带宽: 0.2Hz

    _, h1_01 = calculate_fft(acc_data[:, 1], 200, 0.1)
    _, h2_01 = calculate_fft(acc_data[:, 2], 200, 0.1)
    h_data_01 = np.sqrt(h1_01 ** 2 + h2_01 ** 2)[:data_length // 10]  # 带宽: 0.1Hz

    _, h1_0067 = calculate_fft(acc_data[:, 1], 200, 0.067)
    _, h2_0067 = calculate_fft(acc_data[:, 2], 200, 0.067)
    h_data_0067 = np.sqrt(h1_0067 ** 2 + h2_0067 ** 2)[:data_length // 10]  # 带宽: 0.067Hz

    _, h1_005 = calculate_fft(acc_data[:, 1], 200, 0.05)
    _, h2_005 = calculate_fft(acc_data[:, 2], 200, 0.05)
    h_data_005 = np.sqrt(h1_005 ** 2 + h2_005 ** 2)[:data_length // 10]  # 带宽: 0.05Hz
    # 绘图
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
    # Vertical Component
    line9, = axes[0].plot(v_frc_005, v_data_005, c="black", lw=1.0, linestyle="dotted")
    # dash dot dotted
    line1, = axes[0].plot(v_frc_20, v_data_20, c="blue", lw=1.0, linestyle="solid")
    line8, = axes[0].plot(v_frc_0067, v_data_0067, c="purple", lw=1.0, linestyle="dashed")
    # densely dashed
    line2, = axes[0].plot(v_frc_10, v_data_10, c="green", lw=1.0, linestyle=(0, (5, 1)))
    line7, = axes[0].plot(v_frc_01, v_data_01, c="gray", lw=1.0, linestyle="dashdot")
    # densely dash dotted
    line3, = axes[0].plot(v_frc_08, v_data_08, c="orange", lw=1.0, linestyle=(0, (3, 1, 1, 1)))
    # densely dash dot dotted
    line6, = axes[0].plot(v_frc_02, v_data_02, c="magenta", lw=1.0, linestyle=(0, (3, 1, 1, 1, 1, 1)))
    # loosely dash dotted
    line4, = axes[0].plot(v_frc_067, v_data_067, c="pink", lw=1.0, linestyle=(0, (3, 5, 1, 5, 1, 5)))
    line5, = axes[0].plot(v_frc_04, v_data_04, c="red", lw=2.5, linestyle="solid")
    axes[0].text(0.025, 0.8, "竖向", transform=axes[0].transAxes)
    axes[0].set_xlim([0, max(v_frc_02)])
    axes[0].set_ylim([0, 0.05])
    axes[0].set_ylabel(r"幅值/gal$\cdot$s", labelpad=1)
    axes[0].set_yticks([0, 0.01, 0.02, 0.03, 0.04])
    axes[0].legend((line1, line2, line3, line4, line5, line6, line7, line8, line9),
                   ("窗带宽: 2.0 Hz",
                    "窗带宽: 1.0 Hz",
                    "窗带宽: 0.8 Hz",
                    "窗带宽: 0.67 Hz",
                    "窗带宽: 0.4 Hz",
                    "窗带宽: 0.2 Hz",
                    "窗带宽: 0.1 Hz",
                    "窗带宽: 0.067 Hz",
                    "窗带宽: 0.05 Hz"), framealpha=0.85, title="图例", ncol=3, edgecolor="k")
    # Horizontal Component
    line18, = axes[1].plot(v_frc_005, h_data_005, c="black", lw=1.0, linestyle="dotted")
    line10, = axes[1].plot(v_frc_20, h_data_20, c="blue", lw=1.0, linestyle="solid")
    line17, = axes[1].plot(v_frc_0067, h_data_0067, c="purple", lw=1.0, linestyle="dashed")
    line11, = axes[1].plot(v_frc_10, h_data_10, c="green", lw=1.0, linestyle=(0, (5, 1)))
    line16, = axes[1].plot(v_frc_01, h_data_01, c="gray", lw=1.0, linestyle="dashdot")
    line12, = axes[1].plot(v_frc_08, h_data_08, c="orange", lw=1.0, linestyle=(0, (3, 1, 1, 1)))
    line15, = axes[1].plot(v_frc_02, h_data_02, c="magenta", lw=1.0, linestyle=(0, (3, 1, 1, 1, 1, 1)))
    line13, = axes[1].plot(v_frc_067, h_data_067, c="pink", lw=1.0, linestyle=(0, (3, 5, 1, 5, 1, 5)))
    line14, = axes[1].plot(v_frc_04, h_data_04, c="red", lw=2.5, linestyle="solid")
    axes[1].text(0.025, 0.8, "水平合成向", transform=axes[1].transAxes)
    axes[1].set_xlim([0, max(v_frc_02)])
    axes[1].set_ylim([0, 0.03])
    axes[1].set_ylabel(r"幅值/gal$\cdot$s", labelpad=1)
    axes[1].set_yticks([0, 0.01, 0.02])
    axes[1].set_xlabel("频率/Hz")
    axes[1].legend((line10, line11, line12, line13, line14, line15, line16, line17, line18),
                   ("窗带宽: 2.0 Hz",
                    "窗带宽: 1.0 Hz",
                    "窗带宽: 0.8 Hz",
                    "窗带宽: 0.67 Hz",
                    "窗带宽: 0.4 Hz",
                    "窗带宽: 0.2 Hz",
                    "窗带宽: 0.1 Hz",
                    "窗带宽: 0.067 Hz",
                    "窗带宽: 0.05 Hz"), framealpha=0.85, title="图例", ncol=3, edgecolor="k")
    plt.subplots_adjust(left=0.07, right=0.99, bottom=0.07, top=0.99, hspace=0.05, wspace=0.05)
    os.chdir(r"D:\EarthquakeResearchJournal\1-vectors")
    plt.savefig("fig6b.svg", format="svg")
    os.chdir(r"D:\EarthquakeResearchJournal\3-illustrations")
    plt.savefig("fig6b.jpg", dpi=600)
    # plt.show()


if __name__ == '__main__':
    # import pdb
    # pdb.set_trace()
    plot_figure_six_b()
    os.system("exit")
