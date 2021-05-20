# -*- coding: utf-8 -*-
"""
作者: 王文斌
邮箱：wangwenbineddie@126.com
作用：读取各种地震动数据

"""

import re
import numpy as np

############################################################################

def readKNET(filePath, tag=None):
    """
    作用：读取KNET地震记录fileName的头信息和数据信息。
        地震动数据进行了基线调零并且单位换算成gal。

    参数说明：
        filePath - 文件路径
        tag - 用来标记是读取头信息或者全部信息
            None:默认读取全部信息
            'data':读取全部信息
            'head':只读头信息


    返回值说明：
        tag=None:
        tag='data':
            (data, Long, Lat, Depth, Mag, ,StaCode, StaLong, StaLat, StaH, SampRat, MaxAcc)
        tag='head':
            (Long, Lat, Depth, Mag, StaCode, StaLong, StaLat, StaH, SampRat, MaxAcc)

    """

    fid = open(filePath, 'r')
    for i in range(17):
        line = fid.readline()
        temp = line.split()

        # 读取震源纬度
        if line.startswith('Lat.'):
            Lat = float(temp[-1])

        # 读取震源经度
        if line.startswith('Long.'):
            Long = float(temp[-1])

        # 读取震源深度
        if line.startswith('Depth.'):
            Depth = float(temp[-1])

        # 读取震级
        if line.startswith('Mag.'):
            Mag = float(temp[-1])

        # 读取台站编码
        if line.startswith('Station Code'):
            StaCode = temp[-1]

        # 读取台站纬度
        if line.startswith('Station Lat.'):
            StaLat = float(temp[-1])

        # 读取台站经度
        if line.startswith('Station Long.'):
            StaLong = float(temp[-1])

        # 读取台站高度
        if line.startswith('Station Height'):
            StaH = float(temp[-1])

        # 读取采样率
        if line.startswith('Sampling Freq'):
            temp = re.findall(r'\d.?\d*', temp[-1])
            SampRat = int(temp[0])

        # 读取比例因子
        if line.startswith('Scale Factor'):
            temp = re.findall(r'\d.?\d*', temp[-1])
            ScaFac = float(temp[0]) / float(temp[1])

        # 读取最大加速度值
        if line.startswith('Max. Acc.'):
            MaxAcc = float(temp[-1])

    fid.close()


    # 读取数据信息
    if None == tag or 'data' == tag.lower():
        data = np.genfromtxt(filePath, skip_header=17, skip_footer=1)
        # 使用data.shape = (-1, 1)给矩阵降维出现滤波的错误，要使用ravel函数对矩阵展平。
        data = data.ravel()
        # 地震数据基线调零并且单位转成gal
        data = data - np.mean(data)
        data = data * ScaFac

    reData = dict()
    if None == tag or 'data' == tag.lower():
        reData['data'] = data

    reData['Long'] = Long
    reData['Lat'] = Lat
    reData['Depth'] = Depth
    reData['Mag'] = Mag
    reData['StaCode'] = StaCode
    reData['StaLong'] = StaLong
    reData['StaLat'] = StaLat
    reData['StaH'] = StaH
    reData['SampRat'] = SampRat
    reData['MaxAcc'] = MaxAcc
    return reData

############################################################################

def readSMSD(filePath, tag=None):
    """
    作用：读取中国地震台网地震记录fileName的头信息和数据信息。
        地震动数据进行了基线调零并且单位换算成gal。

    参数说明：
        filePath - 文件路径
        tag - 用来标记是读取头信息或者全部信息
            None:默认读取全部信息
            'data':读取全部信息
            'head':只读头信息

    返回值说明：
        tag=None:
        tag='data':
            (data, Long, Lat, Mag, ,StaCode, StaLong, StaLat, SampRat, MaxAcc, Comp)
        tag='head':
            (Long, Lat, Mag, StaCode, StaLong, StaLat, SampRat, MaxAcc, Comp)

    """

    fid = open(filePath, 'r')
    for i in range(16):
        line = fid.readline()
        temp = line.split()
        # 读取震源纬度/经度/深度
        if line.startswith('EPICENTER'):
            Lat = float(temp[1][:-1])
            Long = float(temp[2][:-1])

        # 读取震级
        if line.startswith('MAGNITUDE'):
            Mag = float(temp[-1][:-4])

        # 读取台站编码/纬度/经度
        if line.startswith('STATION'):
            StaCode = temp[1]
            StaLat = float(temp[2][:-1])
            StaLong = float(temp[3][:-1])

        # 读取采样率
        if line.startswith('NO. OF POINTS'):
            temp = float(temp[-2])
            SampRat = int(1 / temp)

        # 读取最大加速度值
        if line.startswith('PEAK VALUE'):
            MaxAcc = float(temp[2])

        # 读取方向信息
        if line.startswith('COMP'):
            Comp = temp[1]

    fid.close()

    if None == tag or 'data' == tag.lower():
        data = np.genfromtxt(filePath, skip_header=16, skip_footer=1)
        # 使用data.shape = (-1, 1)给矩阵降维出现滤波的错误，要使用ravel函数对矩阵展平。
        data = data.ravel()
        data = data - np.mean(data)

    reData = dict()
    if None == tag or 'data' == tag.lower():
        reData['data'] = data

    reData['Long'] = Long
    reData['Lat'] = Lat
    reData['Mag'] = Mag
    reData['StaCode'] = StaCode
    reData['StaLong'] = StaLong
    reData['StaLat'] = StaLat
    reData['SampRat'] = SampRat
    reData['MaxAcc'] = MaxAcc
    reData['Comp'] = Comp
    return reData

############################################################################

def readPEER(filePath):
    """
    作用：读取PEER网站的地震动数据。
        地震动数据进行了基线调零并且单位换算成gal。

    参数说明：
        filePath - 文件路径

    返回值说明：
        (data, SampRat)

    """
    fid = open(filePath, 'r')
    for i in range(4):
        line = fid.readline()
    fid.close()
    # temp = line.split()
    # SampRat = int(1/float(temp[3]))
    pattern = re.compile(r'[-+]?\d*\.?\d+')
    match = pattern.findall(line)
    SampRat = int(1/float(match[1]))


    data = np.genfromtxt(filePath, skip_header=4, skip_footer=1)
    # 使用data.shape = (-1, 1)给矩阵降维出现滤波的错误，要使用ravel函数对矩阵展平。
    data = data.ravel()
    data = data - np.mean(data)
    data = data * 980

    reData = dict()

    reData['data'] = data
    reData['SampRat'] = SampRat

    return reData