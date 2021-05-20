# -*- coding: utf-8 -*-
"""
作者: 王文斌
邮箱：wangwenbineddie@126.com
作用：地震动P波捡拾

"""

import math
import scipy
import numpy as np

############################################################################

def characteristic_function_p(acc_data):
    """
    :param acc_data:
    :return:
    """
    data_len = len(acc_data)
    cfp = np.zeros(data_len)
    cfp[0] = 2 * acc_data[0] ** 2
    for index in range(1, data_len):
        cfp[index] = acc_data[index] ** 2 + (acc_data[index] - acc_data[index - 1]) ** 2
    # for index in range(1, data_len - 1):
    #     cfp[index] = acc_data[index] ** 2 - acc_data[index - 1] * acc_data[index + 1]
    return cfp

############################################################################

def classic_sta_lta(acc_data, nsta, nlta):
    cfp = characteristic_function_p(acc_data)

    sta = np.cumsum(cfp)

    # Convert to float
    sta = np.require(sta, dtype=np.float)

    # Copy for LTA
    lta = sta.copy()

    # Compute the STA and the LTA
    sta[nsta:] = sta[nsta:] - sta[:-nsta]
    # sta /= nsta
    lta[nlta:] = lta[nlta:] - lta[:-nlta]
    # lta /= nlta

    for i in range(nsta-1, nlta-1):
        lta[i] = lta[i] / (i+1)
    # Pad zeros
    # sta[:nlta - 1] = 0
    sta[:nsta - 1] = lta[:nsta - 1] = 1

    sta[nsta - 1:] /= nsta
    lta[nlta - 1:] /= nlta

    # Avoid division by zero by setting zero values to tiny float
    dtiny = np.finfo(0.0).tiny
    idx = lta < dtiny
    lta[idx] = dtiny

    return sta / lta

############################################################################

def var_aic(data):
    data_len = len(data)
    aic = np.zeros(data_len)
    for index in range(1, data_len - 1):
        aic[index] = (index + 1) * math.log(1 + np.var(data[:index + 1])) + (data_len - index - 1) * math.log(1 + np.var(data[index + 1:]))

    return aic
############################################################################

def polarization_analysis_svd(ud_data, ns_data, ew_data):
    ud_data = ud_data.reshape((-1, 1))
    ns_data = ns_data.reshape((-1, 1))
    ew_data = ew_data.reshape((-1, 1))
    merge_data = np.hstack((ud_data, ns_data, ew_data))
    U, D, V = scipy.linalg.svd(merge_data)

    P_azimuth = math.atan((U[1, 0] * np.sign(U[0, 0]))/(U[2, 0] * np.sign(U[0, 0])))

    P_incidence = math.acos(math.fabs(U[0, 0]))

    P_degree = ((D[0] - D[1])**2 + (D[0] - D[2])**2 + (D[1] - D[2])**2) / np.sum(D) ** 2 / 2

    return P_azimuth, P_incidence, P_degree

############################################################################

def polarization_analysis(ud_acc, ns_acc, ew_acc, win_size):
    data_len = len(ud_acc)
    degree = np.zeros(data_len)
    incidence = np.zeros(data_len)
    azimuth = np.zeros(data_len)

    for i in range(win_size-1, data_len):
        temp = polarization_analysis_svd(ud_acc[i-win_size+1:i+1], ns_acc[i-win_size+1:i+1], ew_acc[i-win_size+1:i+1])
        azimuth[i], incidence[i], degree[i] = temp

    return azimuth, incidence, degree
############################################################################
############################################################################
############################################################################
############################################################################
############################################################################
############################################################################

if __name__ == '__main__':
    pass

