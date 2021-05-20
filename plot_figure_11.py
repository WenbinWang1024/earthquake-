#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# plot_figure_11.py
# Author: Da Zhu
# E-mail: gonglisuozcd@gmail.com
# 2019-12-10 11:22:35
"""
<<地震研究>>期刊投稿图件fig11
"""
import os
import numpy as np
from matplotlib import rcParams
rcParams["axes.unicode_minus"] = False
rcParams["font.size"] = 14.2
rcParams["font.sans-serif"] = "STSong"
rcParams["figure.figsize"] = (5, 4)
rcParams["legend.loc"] = "upper right"
rcParams["legend.fontsize"] = 11
rcParams["legend.handlelength"] = 1.2
rcParams["legend.handletextpad"] = 0.6
import matplotlib.pyplot as plt
from scipy import signal


def bp_filter(data, sampling_freq, low, high):
    nyq = 0.5 * sampling_freq
    low_freq = low / nyq
    high_freq = high / nyq
    b, a = signal.butter(4, [low_freq, high_freq], btype="bandpass")
    data = signal.detrend(data)
    filtered_data = signal.lfilter(b, a, data)
    return filtered_data


def plot_figure_eleven_a():
    file_dir = r"F:\EarthquakeResearchJournal\0-plotting\figure11\data"
    acc_data = np.loadtxt(os.path.join(file_dir, "175-LEA.txt"))
    acc_data_z = acc_data[:, 0]
    acc_data_y = acc_data[:, 1]
    acc_data_x = acc_data[:, 2]
    data_length = len(acc_data_z)
    t = np.linspace(0, data_length / 200, data_length, endpoint=False)
    filtered_z = bp_filter(acc_data_z, 200, 0.3, 10.0)
    filtered_y = bp_filter(acc_data_y, 200, 0.3, 10.0)
    filtered_x = bp_filter(acc_data_x, 200, 0.3, 10.0)
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
    line1, = axes[0].plot(t, acc_data_z, color="k", lw=0.5)
    line2, = axes[0].plot(t, filtered_z, color="r", lw=1)
    axes[0].legend((line1, line2), ("滤波前", "滤波后"))
    axes[0].text(0.05, 0.8, "Z轴", transform=axes[0].transAxes)
    axes[0].set_xlim([0, max(t)])
    axes[0].set_ylabel("加速度/gal", labelpad=2)
    axes[0].set_yticks([-3, 0, 3])

    line3, = axes[1].plot(t, acc_data_y, color="k", lw=0.5)
    line4, = axes[1].plot(t, filtered_y, color="r", lw=1)
    axes[1].legend((line3, line4), ("滤波前", "滤波后"))
    axes[1].text(0.05, 0.8, "Y轴", transform=axes[1].transAxes)
    axes[1].set_xlim([0, max(t)])
    axes[1].set_ylabel("加速度/gal", labelpad=2)

    line5, = axes[2].plot(t, acc_data_x, color="k", lw=0.5)
    line6, = axes[2].plot(t, filtered_x, color="r", lw=1)
    axes[2].legend((line5, line6), ("滤波前", "滤波后"))
    axes[2].text(0.05, 0.8, "X轴", transform=axes[2].transAxes)
    axes[2].set_xlim([0, max(t)])
    axes[2].set_ylabel("加速度/gal", labelpad=2)
    axes[2].set_xlabel("时间/s")
    axes[2].set_xticks([0, 5, 10, 15, 20, 25, 30, 35])
    plt.subplots_adjust(left=0.105, right=0.99, bottom=0.13, top=0.99, hspace=0.15, wspace=0.05)

    # os.chdir(r"F:\EarthquakeResearchJournal\1-vectors")
    # plt.savefig("fig11a.svg", format="svg")

    # os.chdir(r"F:\EarthquakeResearchJournal\3-illustrations")
    # plt.savefig("fig11a.jpg", dpi=600)
    plt.show()


def plot_figure_eleven_b():
    file_dir = r"F:\EarthquakeResearchJournal\0-plotting\figure11\data"
    acc_data = np.loadtxt(os.path.join(file_dir, "181-LSA.txt"))
    acc_data_z = acc_data[:, 0]
    acc_data_y = acc_data[:, 1]
    acc_data_x = acc_data[:, 2]
    data_length = len(acc_data_z)
    t = np.linspace(0, data_length / 200, data_length, endpoint=False)
    filtered_z = bp_filter(acc_data_z, 200, 0.3, 10.0)
    filtered_y = bp_filter(acc_data_y, 200, 0.3, 10.0)
    filtered_x = bp_filter(acc_data_x, 200, 0.3, 10.0)
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
    line1, = axes[0].plot(t, acc_data_z, color="k", lw=0.5)
    line2, = axes[0].plot(t, filtered_z, color="r", lw=1)
    axes[0].legend((line1, line2), ("滤波前", "滤波后"))
    axes[0].text(0.05, 0.8, "Z轴", transform=axes[0].transAxes)
    axes[0].set_xlim([0, max(t)])
    axes[0].set_ylabel("加速度/gal", labelpad=2)

    line3, = axes[1].plot(t, acc_data_y, color="k", lw=0.5)
    line4, = axes[1].plot(t, filtered_y, color="r", lw=1)
    axes[1].legend((line3, line4), ("滤波前", "滤波后"))
    axes[1].text(0.05, 0.8, "Y轴", transform=axes[1].transAxes)
    axes[1].set_xlim([0, max(t)])
    axes[1].set_ylabel("加速度/gal", labelpad=2)

    line5, = axes[2].plot(t, acc_data_x, color="k", lw=0.5)
    line6, = axes[2].plot(t, filtered_x, color="r", lw=1)
    axes[2].legend((line5, line6), ("滤波前", "滤波后"))
    axes[2].text(0.05, 0.8, "X轴", transform=axes[2].transAxes)
    axes[2].set_xlim([0, max(t)])
    axes[2].set_ylabel("加速度/gal", labelpad=2)
    axes[2].set_xlabel("时间/s")
    axes[2].set_yticks([-10, 0, 10])
    axes[2].set_xticks([0, 5, 10, 15, 20, 25, 30])
    plt.subplots_adjust(left=0.12, right=0.975, bottom=0.13, top=0.99, hspace=0.15, wspace=0.05)

    os.chdir(r"F:\EarthquakeResearchJournal\1-vectors")
    plt.savefig("fig11b.svg", format="svg")

    # os.chdir(r"F:\EarthquakeResearchJournal\3-illustrations")
    # plt.savefig("fig11b.jpg", dpi=600)
    # plt.show()


if __name__ == '__main__':
    # import pdb
    # pdb.set_trace()
    plot_figure_eleven_b()
    os.system("exit")
