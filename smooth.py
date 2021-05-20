# -*- coding: utf-8 -*-
"""
作者: 王文斌
邮箱：wangwenbineddie@126.com
作用：利用各种窗对谱进行平滑化
"""

import math
import scipy
import numpy as np


############################################################################

def swin_parzen(power_data, len_time, band):
    """
        作用：根据具有指定带宽的帕曾谱窗，在频域中对给定的功率谱进行平滑化。
            参考《地震动的谱分析入门（第二版）》

        参数说明：
            power_data：numpy.array 地震动的功率谱
            len_time：float 地震动记录的时长
            band: float 帕曾谱窗的带宽(Hz)

        返回值说明：
            smooth_power：numpy.array 平滑化后的功率谱

        """
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
    conv_data = scipy.signal.convolve(power_data, w)
    for i in range(1, lmax):
        conv_data[bp + i] += conv_data[bp - i]
        conv_data[-lmax - i] += conv_data[-lmax + i]

    smooth_power = conv_data[bp:-bp]

    return smooth_power
############################################################################
############################################################################
############################################################################
############################################################################

if __name__ == '__main__':
    pass