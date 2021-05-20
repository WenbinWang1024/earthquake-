# -*- coding: utf-8 -*-
"""
作者: 王文斌
邮箱：wangwenbineddie@126.com
作用：地震数据处理
"""

import math
import numba as nb
import numpy as np
import scipy.signal as scipySingal


############################################################################
# 使用nb.jit后加速效果明显，可以提高大约二百五十倍的速度
@nb.jit
def getV_D(Acc, SampRat):
    """
    作用：根据加速度的时间历程，计算速度和位移时间历程。
        参考《地震动的谱分析入门（第二版）》
        
    参数说明：
        Acc：存储地震动加速度值
        SampRat：地震动记录的采样率，单位HZ
        
    返回值说明：
        V：地震动速度记录
        D：地震动位移记录
    
    """
    dt = 1.0 / float(SampRat)
    dataLen = len(Acc)
    V = np.zeros(dataLen)
    D = np.zeros(dataLen)

    for i in range(1, dataLen):
        V[i] = V[i - 1] + (Acc[i] + Acc[i - 1]) * dt / 2
        D[i] = D[i - 1] + V[i - 1] * dt + (Acc[i - 1] / 3 + Acc[i] / 6) * dt ** 2
    # D[i] = D[i-1] + (V[i] + V[i-1])*dt/2

    return V, D


############################################################################
# 使用nb.jit后加速效果明显，可以提高大约二百五十倍的速度
@nb.jit
def getV_D_yu(Acc, SampRat):
    """
    作用：根据加速度的时间历程，计算速度和位移时间历程。
        根据于海英老师提出的积分以前必须基线调零修改的积分方法

    参数说明：
        Acc：存储地震动加速度值
        SampRat：地震动记录的采样率，单位HZ

    返回值说明：
        V：地震动速度记录
        D：地震动位移记录

    """
    dt = 1.0 / float(SampRat)
    dataLen = len(Acc)
    V = np.zeros(dataLen)
    D = np.zeros(dataLen)

    # 加速度基线调零
    Acc = Acc - np.mean(Acc)

    # 计算速度
    for i in range(1, dataLen):
        V[i] = V[i - 1] + (Acc[i] + Acc[i - 1]) * dt / 2

    # 速度基线调零
    V = V - np.mean(V)

    # 计算位移
    for i in range(1, dataLen):
        D[i] = D[i - 1] + (V[i] + V[i - 1]) * dt / 2

    # 位移基线调零
    D = D - np.mean(D)

    return V, D


############################################################################
# 使用nb.jit后加速效果明显，可以提高大约二百八十倍的速度
@nb.jit
def CRAC(Acc, Vel, Dis, SampRat):
    dataLen = len(Acc)
    dt = 1.0 / float(SampRat)
    T = dt * dataLen

    temp = np.zeros(dataLen)
    for i in range(dataLen):
        t = i * dt
        temp[i] = Dis[i] * (3 * T * t ** 2 - 2 * t ** 3)

    interSum = 0
    for i in range(1, dataLen):
        interSum = interSum + (temp[i] + temp[i - 1]) * dt / 2

    a1 = 28 / 13 / T ** 2 * (2 * Vel[-1] - 15 * interSum / T ** 5)
    a0 = Vel[-1] / T - a1 * T / 2

    corAcc = np.zeros(dataLen)
    corVel = np.zeros(dataLen)
    corDis = np.zeros(dataLen)

    for i in range(dataLen):
        t = i * dt
        corAcc[i] = Acc[i] - (a0 + a1 * t)
        corVel[i] = Vel[i] - (a0 * t + a1 * t ** 2 / 2)
        corDis[i] = Dis[i] - (a0 * t ** 2 / 2 + a1 * t ** 3 / 6)

    corAcc = corAcc * np.max(np.abs(Acc)) / np.max(np.abs(corAcc))
    return corAcc, corVel, corDis


############################################################################
# 使用nb.jit后不能起到加速效果，运行速度反而稍稍降低
# @nb.jit
def getFFT(data, sampRat):
    """
    作用：计算地震动的傅里叶振幅谱
        参考《地震动的谱分析入门》大崎顺彦
        
    参数说明：
        data：需要进行傅里叶变换的数据，此处一般是地震动加速度记录
        sampRat：数据的采样率
        
    返回值说明：
        frc：频率，只有sampRat的一半
        fftData：傅里叶振幅谱  
    
    """
    dataLen = len(data)
    frc = np.linspace(0, sampRat, dataLen, endpoint=False)

    fftData = np.fft.fft(data)
    fftData = np.abs(fftData) / sampRat
    frc = frc[:int(dataLen / 2) + 1]
    fftData = fftData[:int(dataLen / 2) + 1]
    return frc, fftData


############################################################################

def getPower(fft_data, len_time):
    """
    作用：根据地震动的傅氏谱，计算功率谱。
        参考《地震动的谱分析入门》大崎顺彦

    参数说明：
        fft_data：numpy.array 地震动的傅氏谱
        len_time: float 地震动的时长

    返回值说明：
        power：numpy.array 地震动的功率谱

    """
    power_data = 2 * fft_data ** 2 / len_time
    power_data[0] = fft_data[0] ** 2 / len_time
    power_data[-1] = fft_data[-1] ** 2 / len_time
    return power_data


############################################################################

def power2fft(power_data, len_time):
    """
    作用：根据地震动的功率谱，计算傅氏谱。
        参考《地震动的谱分析入门》大崎顺彦

    参数说明：
        power_data：numpy.array 地震动的功率谱
        len_time: float 地震动的时长

    返回值说明：
        fft_data：numpy.array 地震动的傅氏谱

    """
    fft_data = np.sqrt(power_data * len_time / 2)
    fft_data[0] = math.sqrt(power_data[0] * len_time)
    fft_data[-1] = math.sqrt(power_data[-1] * len_time)
    return fft_data


############################################################################
# 使用nb.jit后不能起到加速效果，运行速度反而稍稍降低
# @nb.jit
def butterBand(data, lowCut, highCut, sampRat, order=4):
    """
    作用：计算巴特沃斯通带滤波
    
    参数说明：
        data：需要进行滤波的数据，此处一般是地震动加速度记录
        lowCut：低频截止频率
        highCut：高频截止频率
        sampRat：数据的采样频率HZ
        order：巴特沃斯滤波的阶数
        
    返回值说明：
        butterData：经过巴特沃斯通带滤波后的数据
        
    """
    f0 = 0.5 * sampRat
    low = lowCut / f0
    high = highCut / f0
    b, a = scipySingal.butter(order, [low, high], btype='bandpass')
    butterData = scipySingal.lfilter(b, a, data)
    return butterData


############################################################################
# 使用nb.jit后不能起到加速效果，运行速度反而大大降低
# @nb.jit
def getIa(eqAc, sampRat):
    """
    作用：计算阿里亚斯强度Ia
    
    参数说明：
        eqAc：地震动加速度记录
        sampRat：地震动记录采样率
    
    返回值说明：
        Ia：阿里亚斯强度
    
    """
    dt = 1 / sampRat
    eqAc = eqAc ** 2

    # 利用sum进行梯形法则积分
    temp = np.sum(eqAc) + np.sum(eqAc[1:-1])
    temp = temp * dt / 2

    Ia = np.pi * temp / 2 / 9.8
    return Ia


############################################################################
# 使用nb.jit后不能起到加速效果，运行速度反而大大降低
# @nb.jit
def getCAV(eqAc, sampRat):
    """
    作用：计算累积绝对速度CAV
    
    参数说明：
        eqAc：地震动加速度记录
        sampRat：地震动记录采样率
        
    返回值说明：
        CAV：累积绝对速度
    
    """
    eqAc = np.abs(eqAc)
    # 利用sum进行梯形法则积分
    temp = np.sum(eqAc) + np.sum(eqAc[1:-1])
    CAV = temp / sampRat / 2
    return CAV


############################################################################
# 使用nb.jit后加速效果明显，运行速度大大提高
@nb.jit
def getCAVstd(eqAc, sampRat):
    """
    作用：计算标准累积绝对速度CAVstd

    参数说明：
        eqAc：地震动加速度记录
        sampRat：地震动记录采样率

    返回值说明：
        CAVstd：标准累积绝对速度

    """
    eqAc = np.abs(eqAc)
    dataLen = len(eqAc)
    stepNums = math.ceil(dataLen / sampRat)

    for stepNum in range(stepNums):
        beginPoint = stepNum * sampRat
        endPoint = beginPoint + sampRat
        if 25 > np.max(eqAc[beginPoint:endPoint]):
            eqAc[beginPoint:endPoint] = 0.0

    # 利用sum进行梯形法则积分
    temp = np.sum(eqAc) + np.sum(eqAc[1:-1])
    CAVstd = temp / sampRat / 2
    return CAVstd


############################################################################
# 使用nb.jit后不能起到加速效果，运行速度反而大大降低
# @nb.jit
def getCAV5(eqAc, sampRat):
    """
    作用：计算CAV5

    参数说明：
        eqAc：地震动加速度记录
        sampRat：地震动记录采样率

    返回值说明：
        CAV5：CAV5

    """
    eqAc = np.abs(eqAc)
    eqAc = np.where(eqAc >= 5, eqAc, 0)

    # 利用sum进行梯形法则积分
    temp = np.sum(eqAc) + np.sum(eqAc[1:-1])
    CAV5 = temp / sampRat / 2
    return CAV5


############################################################################
# 使用nb.jit后不能起到加速效果，运行速度反而大大降低
# @nb.jit
def getIV2(eqVe, sampRat):
    """
    作用：计算速度平方积分IV2
    
    参数说明：
        eqVe：竖向地震动速度
        sampRat：地震动记录的采样率
        
    返回值说明：
        IV2：速度平方积分
    
    """
    eqVe = eqVe ** 2
    # 利用sum进行梯形法则积分
    temp = np.sum(eqVe) + np.sum(eqVe[1:-1])
    IV2 = temp / sampRat / 2
    return IV2


############################################################################
# 使用nb.jit后加速效果明显，可以提高大约一百倍的速度
@nb.jit
def getTe(eqAc, sampRat, percent=0.9):
    """
    作用：计算相对能量持时Te
    
    参数说明：
        eqAc：地震动加速度记录
        sampRat：地震动记录的采样率
        percent：判断是90%能量持时还是70%能量持时
        
    返回值说明：
        Te：相对能量持时
        
    """
    dataLen = len(eqAc)
    lowCut = (1 - percent) / 2
    hightCut = 1 - lowCut
    eqAc = eqAc ** 2
    temp = np.zeros(dataLen)
    for i in range(1, dataLen):
        temp[i] = temp[i - 1] + (eqAc[i] + eqAc[i - 1]) / sampRat / 2

    temp = temp / temp[-1]

    lowPoint = 0
    hightPoint = 0
    for i in range(dataLen):
        if temp[i] >= lowCut:
            lowPoint = i
            break
    for i in range(dataLen):
        if temp[i] >= hightCut:
            hightPoint = i
            break

    Te = (hightPoint - lowPoint) / sampRat
    return Te


############################################################################
# 使用nb.jit后不能起到加速效果，运行速度反而降低一百五十倍
# @nb.jit
def getArms(eqAc, sampRat):
    """
    作用：计算均方根加速度Arms
    
    参数说明：
        eqAc：地震动加速度记录
        sampRat：地震动记录的采样率
        
    返回值说明：
        Arms：均方根加速度
        
    """
    dataLen = len(eqAc)
    T = dataLen / sampRat
    eqAc = eqAc ** 2
    # 利用sum进行梯形法则积分
    temp = np.sum(eqAc) + np.sum(eqAc[1:-1])
    temp = temp / sampRat / 2
    temp = temp / T
    Arms = np.sqrt(temp)
    return Arms


############################################################################
# 使用nb.jit后加速效果明显，可以提高大约一百倍的速度
@nb.jit
def getRes(ACX, sampRat, d):
    """
        作用：计算加速度、速度和位移反应谱

        参数说明：
            ACX：numpy.array 加速度记录
            sampRat：加速度采样率
            d: 阻尼比

        返回值说明：
            ACmax： numpy.array 加速度反应谱
            VEmax： numpy.array 速度反应谱
            DImax： numpy.array 位移反应谱
            Ta：numpy.array 周期

        """
    t = 1 / float(sampRat)

    Ta = np.linspace(0.04, 6, 597)

    dataLen = len(ACX)
    pLen = len(Ta)

    ACmax = np.zeros(pLen)
    VEmax = np.zeros(pLen)
    DImax = np.zeros(pLen)

    DI = np.zeros(dataLen)
    VE = np.zeros(dataLen)
    AC = np.zeros(dataLen)

    for j, T1 in enumerate(Ta):
        w0 = 2 * np.pi / T1
        U1 = (1 / (w0 * np.sqrt(1 - d ** 2))) * np.sin((w0 * np.sqrt(1 - d ** 2)) * t)
        U2 = np.cos(w0 * np.sqrt(1 - d ** 2) * t)
        v = np.exp(-d * w0 * t)
        r1 = (2 * (d ** 2) - 1) / (w0 ** 2 * t)
        r2 = 2 * d / (w0 ** 3 * t)
        r3 = d / (w0 ** 2 * t)
        r4 = d / w0
        k1 = w0 ** 2 * (1 - d ** 2) * U1 + d * w0 * U2
        k2 = r2 + r3 * t / d
        k3 = U2 - d * w0 * U1
        k4 = r1 + r4
        a1 = v * (d * w0 * U1 + U2)
        a2 = v * U1
        a3 = -w0 ** 2 * (v * U1)
        a4 = v * k3
        b1 = v * (U1 * k4 + U2 * k2) - r2
        b2 = -v * (U1 * r1 + U2 * r2) - t * r3 / d + r2
        b3 = v * (k3 * k4 - k1 * k2) + r3 / d
        b4 = -v * (r1 * k3 - r2 * k1) - r3 / d
        DI[0] = 0
        VE[0] = -ACX[0] * t
        AC[0] = 2 * d * w0 * ACX[0] * t
        for i in range(1, dataLen - 1):
            DI[i + 1] = a1 * DI[i] + a2 * VE[i] + b1 * ACX[i] + b2 * ACX[i + 1]
            VE[i + 1] = a3 * DI[i] + a4 * VE[i] + b3 * ACX[i] + b4 * ACX[i + 1]
            AC[i + 1] = -2 * d * w0 * VE[i + 1] - w0 ** 2 * DI[i + 1]

        ACmax[j] = np.max(np.abs(AC))
        VEmax[j] = np.max(np.abs(VE))
        DImax[j] = np.max(np.abs(DI))

    return ACmax, VEmax, DImax, Ta


############################################################################
# 使用nb.jit后，10000000次循环后加速效果明显，大概提高十倍。
# 但是只有少量循环后，反而大大降低了运行效率，所以建议不使用nb.jit加速
# @nb.jit
def getSIh(period, Sv):
    """
    作用：计算Housner定义的谱强度
    
    参数说明：
        period：周期
        Sv：相对速度反应谱
    
    返回值说明：
        SIh：谱强度
    
    """
    beginPoint, endPoint = np.searchsorted(period, [0.1, 2.5])
    Sv = Sv[beginPoint:(endPoint + 1)]

    # 积分
    temp = np.sum(Sv) + np.sum(Sv[1:-1])
    SIh = temp * (period[1] - period[0]) / 2
    return SIh


############################################################################
# 使用nb.jit后，10000000次循环后加速效果明显，大概提高十倍。
# 但是只有少量循环后，反而大大降低了运行效率，所以建议不使用nb.jit加速
# @nb.jit
def getSIc(period, Sv):
    """
    作用：计算Clough定义的谱强度
    
    参数说明：
        period：周期
        Sv：相对速度反应谱
    
    返回值说明：
        SIc：谱强度
    
    """
    beginPoint, endPoint = np.searchsorted(period, [0.1, 1.0])
    Sv = Sv[beginPoint:(endPoint + 1)]

    # 积分
    temp = np.sum(Sv) + np.sum(Sv[1:-1])
    SIc = temp * (period[1] - period[0]) / 2
    return SIc


############################################################################
# 使用nb.jit后，不能起到加速效果，反而大大降低了运行速度
# @nb.jit
def __getAbsMax(data):
    """
    作用：获取地震动时程的峰值
    
    参数说明：
        data：地震动时程
        
    返回值说明：
        absMax：地震动峰值
    
    """
    maxIndex = np.argmax(np.abs(data))
    absMax = data[maxIndex]
    return absMax


############################################################################
# 使用nb.jit后，不能起到加速效果，反而大大降低了运行速度
# @nb.jit
def getPGA(eqAc):
    """
    作用：获取地震动加速度时程的峰值PGA
    
    参数说明：
        eqAc：地震动加速度时程
        
    返回值说明：
        PGA：地震动加速度峰值
    
    """

    PGA = __getAbsMax(eqAc)
    return PGA


############################################################################
# 使用nb.jit后，不能起到加速效果，反而大大降低了运行速度
# @nb.jit
def getPGV(eqVe):
    """
    作用：获取地震动速度时程的峰值PGV
    
    参数说明：
        eqVe：地震动速度时程
        
    返回值说明：
        PGV：地震动速度峰值
    
    """

    PGV = __getAbsMax(eqVe)
    return PGV


############################################################################
# 使用nb.jit后，不能起到加速效果，反而大大降低了运行速度
# @nb.jit
def getPGD(eqDi):
    """
    作用：获取地震动位移时程的峰值PGV
    
    参数说明：
        eqDi：地震动位移时程
        
    返回值说明：
        PGD：地震动位移峰值
    
    """

    PGD = __getAbsMax(eqDi)
    return PGD


############################################################################
# 使用nb.jit后，不能起到加速效果，反而大大降低了运行速度
# @nb.jit
def getI(eqAc1, eqVe1, eqAc2, eqVe2, eqAc3, eqVe3):
    """
    作用：计算三通道的仪器地震烈度
    
    参数说明：
        eqAc1：numpy.array 第一通道的加速度数据，单位为gal
        eqVe1：numpy.array 第一通道的速度数据
        eqAc2：numpy.array 第二通道的加速度数据
        eqVe2：numpy.array 第二通道的速度数据
        eqAc3：numpy.array 第三通道的加速度数据
        eqVe3：numpy.array 第三通道的速度数据
    
    返回值说明：
        I：string 仪器地震烈度
    
    """

    dataLen = len(eqAc1)

    RA = np.zeros(dataLen)
    RV = np.zeros(dataLen)

    # 三通道数据合成
    RA = np.sqrt(eqAc1 ** 2 + eqAc2 ** 2 + eqAc3 ** 2)
    RV = np.sqrt(eqVe1 ** 2 + eqVe2 ** 2 + eqVe3 ** 2)

    # 计算地震动加速度峰值和速度峰值
    # 注意：需要进行单位换算为m/s/s
    PGA = np.max(RA) / 100.0
    PGV = np.max(RV) / 100.0

    # 计算仪器地震烈度
    Ipga = 3.17 * np.log10(PGA) + 6.59
    Ipgv = 3.00 * np.log10(PGV) + 9.77

    # 仪器地震烈度取值
    if Ipga >= 6.0 and Ipgv >= 6.0:
        I = Ipgv
    else:
        I = (Ipga + Ipgv) / 2

    if I < 1.0:
        I = 1.0

    if I > 12.0:
        I = 12.0

    I = '%.1f' % I

    return I


############################################################################
# 使用nb.jit后，不能起到加速效果，反而大大降低了运行速度
# @nb.jit
def getPSA(Sa):
    """
    作用：计算加速度反应谱峰值
    
    参数说明：
        Sa：numpy.array 加速度反应谱
    
    返回值说明：
        PSA：float 加速度反应谱峰值
    
    """
    PSA = np.max(Sa)

    return PSA


############################################################################
# 使用nb.jit后，不能起到加速效果，反而大大降低了运行速度
# @nb.jit
def getPSV(Sv):
    """
    作用：计算速度反应谱峰值

    参数说明：
        Sv：numpy.array 速度反应谱

    返回值说明：
        PSV：float 速度反应谱峰值

    """
    PSV = np.max(Sv)

    return PSV


############################################################################
@nb.jit
def getEPA(period, SA):
    """
    作用：计算有效峰值加速度

    参数说明：
        period：周期
        SA：numpy.array 加速度反应谱

    返回值说明：
        EPA：float 有效峰值加速度

    """
    beginPoint, endPoint = np.searchsorted(period, [0.1, 0.5])
    SA = SA[beginPoint:(endPoint + 1)]

    Sa = np.mean(SA)

    EPA = Sa / 2.5

    return EPA


############################################################################
@nb.jit
def getEPV(period, SV):
    """
    作用：计算有效峰值速度

    参数说明：
        period：周期
        SV：numpy.array 速度反应谱

    返回值说明：
        EPV：float 有效峰值速度

    """
    beginPoint, endPoint = np.searchsorted(period, [0.8, 1.2])
    SV = SV[beginPoint:(endPoint + 1)]

    Sv = np.mean(SV)

    EPV = Sv / 2.5

    return EPV


############################################################################
# 使用nb.jit后，程序会出错
# @nb.jit
def getLowCut(Acc, SampRat, hp=35):
    data_len = len(Acc)

    lps = np.linspace(0.01, 0.5, 50)
    lps = lps[lps >= 2 * SampRat / data_len]

    t_all = np.linspace(0, data_len / SampRat, data_len, endpoint=False)

    # 四分之一长度数据作为尾部长度
    last_len = data_len // 4

    # 存储位移尾部的线性回归的斜率
    slopes = np.zeros(len(lps))
    # 存储PGD
    pgds = np.zeros(len(lps))
    # 存储位移尾部的均值
    dis_last = np.zeros(len(lps))

    # butterworth带通滤波器
    for i, lp in enumerate(lps):

        acc_after = butterBand(Acc, lp, hp, SampRat, 4)
        vel_after, dis_after = getV_D(acc_after, SampRat)

        dis_after_last_data = dis_after[-last_len:]
        dis_after_last_data = dis_after_last_data - np.mean(dis_after_last_data)

        pgds[i] = math.fabs(getPGD(dis_after))
        dis_last[i] = np.mean(np.abs(dis_after_last_data))

        # 线性回归，不难得到直线的斜率、截距
        slope = np.polyfit(t_all[-last_len:], dis_after_last_data, 1)

        slopes[i] = slope[0]

        if dis_last[i] < pgds[i] / 4 and math.fabs(slopes[i]) < pgds[i] / 440:
            return lp

    return 1


############################################################################
# 使用nb.jit后，程序会出错
# @nb.jit
def getLowCut_zhang(Acc, SampRat):
    hp = 35
    lps = [i * 0.01 + 0.04 for i in range(97)]
    lps.append(0.067)
    lps = sorted(lps)
    data_len = len(Acc)
    t_all = np.linspace(0, data_len / SampRat, data_len, endpoint=False)

    # 存储位移后10s的线性回归的斜率
    slopes = np.zeros(len(lps))
    # 存储PGD
    pgds = np.zeros(len(lps))
    # 存储位移后10秒的均值
    d_last40s = np.zeros(len(lps))

    # butterworth带通滤波器
    for i, lp in enumerate(lps):

        all_after = butterBand(Acc, lp, hp, SampRat, 4)
        v, d = getV_D(all_after, SampRat)

        d_max = getPGD(d)
        pgds[i] = math.fabs(d_max)
        d_last40s[i] = np.mean(np.abs(d[(data_len - SampRat * 10):]))

        # 线性回归，不难得到直线的斜率、截距
        slope = np.polyfit(t_all[(data_len - SampRat * 10):].flatten(), d[(data_len - SampRat * 10):].flatten(), 1)

        slopes[i] = slope[0]

        if i > 0:
            if (d_last40s[i] < 0.1 and d_last40s[i] < pgds[i] / 10) and math.fabs(slopes[i]) < pgds[i] / 1000:
                return lp

    return 0


############################################################################


if __name__ == '__main__':
    pass
    #    import matplotlib.pyplot as plt
    #    import math
    # import time
    #
    # eq = np.random.rand(20000)
    # a = np.ones(20000)
    # a = getArms(eq, 100)
    # print(a)
# bT = time.clock()
#    for i in range(1000):
#        b = eq ** 2
#        
#    print(time.clock() - bT)

#    data = readKNET("IWT0121103110314.EW", 'data')
#    V, D = getV_D(data, 100)
#    A, V, D = CRAC(data, V, D, 100)
##    V, D = getV_D(A, 100)
#    x = np.linspace(0, 60, 6000, endpoint=False)
#    plt.figure()
#    plt.plot(x, D)
##    plt.show()
#    plt.savefig('text.jpg')
#    sampRat = 100
#    dt = 1/sampRat
#    T = 10
#
#    t = np.linspace(0, T, T*sampRat, endpoint=False)
#    y = 5*np.sin(2*math.pi*t*5)+10*np.sin(2*math.pi*t*25)
#    
#    data = butterBand(y, 0.01, 15, sampRat, 4)
#    f, ff = getFFT(data, sampRat)
#    print(len(ff))
#    plt.figure()
#    plt.plot(f, ff)
#    plt.show()
