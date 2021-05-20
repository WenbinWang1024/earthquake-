# -*- coding: utf-8 -*-
"""
作者: 王文斌
邮箱：wangwenbineddie@126.com
作用：计算和地理信息相关的模块

"""
import math


def getDistance(Lat_A, Lng_A, Lat_B, Lng_B):
    """
        作用：根据地球两点的经纬度信息，计算球面距离

        参数说明：
            Lat_A - A点的纬度
            Lng_A - A点的经度
            Lat_B - B点的纬度
            Lng_B - B点的经度

        返回值说明：
            distance - 两点间的球面距离，单位km

        """
    ra = 6378.137  # 赤道半径
    rb = 6356.752  # 极半径 （km）
    flatten = (ra - rb) / ra  # 地球偏率
    rad_lat_A = math.radians(Lat_A)
    rad_lng_A = math.radians(Lng_A)
    rad_lat_B = math.radians(Lat_B)
    rad_lng_B = math.radians(Lng_B)
    pA = math.atan(rb / ra * math.tan(rad_lat_A))
    pB = math.atan(rb / ra * math.tan(rad_lat_B))
    xx = math.acos(math.sin(pA) * math.sin(pB) + math.cos(pA) * math.cos(pB) * math.cos(rad_lng_A - rad_lng_B))
    c1 = (math.sin(xx) - xx) * (math.sin(pA) + math.sin(pB)) ** 2 / math.cos(xx / 2) ** 2
    c2 = (math.sin(xx) + xx) * (math.sin(pA) - math.sin(pB)) ** 2 / math.sin(xx / 2) ** 2
    dr = flatten / 8 * (c1 - c2)
    distance = ra * (xx + dr)
    return distance

