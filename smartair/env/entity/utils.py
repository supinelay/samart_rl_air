import math

import numpy as np


def func(x, y, angle, v, az):
    dx = v * math.cos(angle)
    dy = v * math.sin(angle)
    dangle = az / v
    return dx, dy, dangle

def rk_4(x, y, angle, v, az, delta):
    X_1, Y_1, Angle_1 = func(x, y, angle, v, az)

    X_2, Y_2, Angle_2 = func(x + delta / 2 * X_1, y + delta / 2 * Y_1, angle + delta / 2 * Angle_1, v, az)
    X_3, Y_3, Angle_3 = func(x + delta / 2 * X_2, y + delta / 2 * Y_2, angle + delta / 2 * Angle_2, v, az)
    X_4, Y_4, Angle_4 = func(x + delta * X_3, y + delta * Y_3, angle + delta * Angle_3, v, az)

    delta_x = (X_1 + 2 * X_2 + 2 * X_3 + X_4) * delta / 6
    delta_y = (Y_1 + 2 * Y_2 + 2 * Y_3 + Y_4) * delta / 6
    delta_angle = (Angle_1 + 2 * Angle_2 + 2 * Angle_3 + Angle_4) * delta / 6
    return delta_x, delta_y, delta_angle, az / v

def transform_coordinate(lon, lat):
    x = lon
    y = lat
    return x,y

def lat_lon_to_xyz(lat_rad, lon_rad, r=6371):
    x = r * math.cos(lat_rad) * math.cos(lon_rad)
    y = r * math.cos(lat_rad) * math.sin(lon_rad)
    z = r * math.sin(lat_rad)
    return x,y,z


def xyz_to_wgs84(x, y, z):
   # a = 6378137.0  # WGS84椭球体长半轴
   # f = 1 / 298.257223563  # WGS84椭球体扁率
  #  b = a * (1-f)       # WGS84椭球体短半轴
  #  c = math.sqrt((a**2 - b**2) / (a**2))   # WGS84椭球体第一偏心率平方
   # d = math.sqrt((a**2 - b**2) / (b**2))
    # 计算经度
    #lon = math.atan2(y, x)
    # 迭代计算维度和高度
  #  p = math.sqrt(x ** 2 + y ** 2)
    #theta = math.atan2(z * a, p * b)

   # lat = math.atan2((z + (d**2) * b * np.power(np.sin(theta), 3)),
    #                 (p - (c**2) * a * np.power(np.cos(theta), 3)))

   # N = a / math.sqrt(1 - (c**2) * math.sin(lat) ** 2)
   # h = p / math.cos(lat) - N

#lf地心直角坐标系to地理坐标系
    d_pi=3.141592653589793
    earth_a=6378137.0  #半长轴
    earth_f=1/298.257223563 #地球扁率
    earth_b=(1-earth_f)*earth_a#地球短半径r

    earth_e2=(earth_a*earth_a-earth_b*earth_b)/(earth_a*earth_a)#地球第一偏心率的平方
    earth_ep2=(earth_a*earth_a-earth_b*earth_b)/(earth_b*earth_b)#地球第二偏心率的平方


    p=math.sqrt(x**2+y**2)
    angle_x=(z*earth_a)/(p*earth_b)
    theta=math.atan((z*earth_a)/(p*earth_b))
    lat=math.atan((z+earth_ep2*earth_b*(math.sin(theta))**3)/(p-earth_e2*earth_a*(math.cos(theta))**3))
    N_2=earth_a/(math.sqrt(1-earth_e2*(math.sin(lat)**2)))
    h=p/math.cos(lat)-N_2

    if y<0:
        lon=math.atan2(y,x)+2*d_pi
    else:
        lon=math.atan2(y,x)
    # lat = math.degrees(lat)
    # lon = math.degrees(lon)

    return lat, lon, h

# 84坐标系转xyz
def wgs84_to_xyz(lat_rad, lon_rad, h):

   #a = 6378137.0  # WGS84椭球体长半轴
  # f = 1 / 298.257223563  # WGS84椭球体扁率
  # b = a * (1 - f)  # WGS84椭球体长半轴
  # e_sq = f * (2 - f)  # WGS84椭球体第一偏心率平方
  # e_2 = (a*a -b*b) /(a*a)

   #lat_rad = math.radians(lat)
  ## lon_rad = math.radians(lon)

    # 计算N值
   #N = a / math.sqrt(1 - e_sq * math.sin(lat_rad) ** 2)

    # 计算xyz坐标
  # x = (N + h) * math.cos(lat_rad) * math.cos(lon_rad)
  # y = (N + h) * math.cos(lat_rad) * math.sin(lon_rad)
   #z = (N * (1-e_sq) + h) * math.sin(lat_rad)

#lf修正84to地心直角坐标系
    earth_a=6378137.0  #半长轴
    earth_f=1/298.257223563 #地球扁率
    earth_b=(1-earth_f)*earth_a#地球短半径r

    earth_e2=(earth_a*earth_a-earth_b*earth_b)/(earth_a*earth_a)#地球第一偏心率的平方
    earth_ep2=(earth_a*earth_a-earth_b*earth_b)/(earth_b*earth_b)#地球第二偏心率的平方
    N_1=earth_a/(math.sqrt(1-earth_e2*(math.sin(lat_rad))**2))

    x=(N_1+h)*math.cos(lat_rad)*math.cos(lon_rad)
    y=(N_1+h)*math.cos(lat_rad)*math.sin(lon_rad)
    z=((1-earth_e2)*N_1+h)*math.sin(lat_rad)
    return x, y, z

# xyz坐标系到84坐标系的速度转化
def vec_xyz_84(x, y, z, v_xyz):

    a = 6378137.0  # WGS84椭球体长半轴
    f = 1 / 298.257223563  # WGS84椭球体扁率
    b = a * (1 - f)  # WGS84椭球体长半轴
    e_sq = 2 * f - f ** 2  # WGS84椭球体第一偏心率平方

    N = a / np.sqrt(1 - e_sq * np.sin(y) ** 2)

    R = np.array([
        [-np.sin(x), np.cos(x), 0],
        [-np.sin(y) * np.cos(x), -np.sin(y) * np.sin(x), np.cos(y)],
        [np.cos(y) * np.cos(x), np.cos(y) * np.sin(x), np.sin(y)]
    ])

    v_wgs84 = np.dot(R, v_xyz)

    return v_wgs84

def enu_to_ecef(xEast, yNorth, zUp, lat0, lon0, h0):

    a = 6378137.0
    f = 1 / 298.257223563  # WGS84椭球体扁率
    e_sq = f * (2 - f)
    pi = 3.14159265359

    lamb = pi / 180 * (lat0)
    phi = pi / 180 * (lon0)
    s = np.sin(lamb)
    N = a / np.sqrt(1 - e_sq * s * s)

    sin_lambda = np.sin(lamb)
    cos_lambda = np.cos(lamb)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    x0 = (h0 + N) * cos_lambda * cos_phi
    y0 = (h0 + N) * cos_lambda * sin_phi
    z0 = (h0 + (1 - e_sq) * N) * sin_lambda

    t = cos_lambda * zUp - sin_lambda * yNorth

    zd = sin_lambda * zUp + cos_lambda * yNorth
    xd = cos_phi * t - sin_phi * xEast
    yd = sin_phi * t + cos_phi * xEast

    x = xd + x0
    y = yd + y0
    z = zd + z0
    return x, y, z

def var(list1):
    if len(list1) == 0:
        return 0.0
    # 计算平均值
    mean = sum(list1)/len(list1)

    diff_list = [(x - mean)**2 for x in list1]

    return sum(diff_list)/ len(diff_list)





# def f1(x, y, v, angle, acc, omega):
#     """
#
#     :param t: 时间
#     :param x: x坐标
#     :param y: y坐标
#     :param v: 速度
#     :param puxi: 航向角
#     :param a: 加速度
#     :param oumiga:角速度
#     :return:
#     """
#     df = v * math.cos(angle)
#     return df
#
#
# def f2(x, y, v, angle, a, omega):
#     """
#
#     :param t: 时间
#     :param x: x坐标
#     :param y: y坐标
#     :param v: 速度
#     :param puxi: 航向角
#     :param a: 加速度
#     :param oumiga:角速度
#     :return:
#     """
#     # angle = math.radians(angle)
#     df = v * math.sin(angle)
#     return df
#
#
# def f3(x, y, v, angle, acc, omega):
#     """
#
#     :param t: 时间
#     :param x: x坐标
#     :param y: y坐标
#     :param v: 速度
#     :param puxi: 航向角
#     :param a: 加速度
#     :param oumiga:角速度
#     :return:
#     """
#     df = acc
#     return df
#
#
# def f4(x, y, v, angle, acc, omega):
#     """
#
#     :param t: 时间
#     :param x: x坐标
#     :param y: y坐标
#     :param v: 速度
#     :param puxi: 航向角
#     :param a: 加速度
#     :param oumiga:角速度
#     :return:
#     """
#     df = omega/v
#     return df

# def compute_next(x, y, v, angle, acc, omega,delta_t):
#     X_1 = f1(x, y, v, angle, acc, omega)
#     Y_1 = f2(x, y, v, angle, acc, omega)
#     V_1 = f3(x, y, v, angle, acc, omega)
#     P_1 = f4(x, y, v, angle, acc, omega)
#
#     X_2 = f1(x + delta_t / 2 * X_1,
#              y + delta_t / 2 * Y_1,
#              v + delta_t / 2 * V_1,
#              angle + delta_t / 2 * P_1,
#              acc,
#              omega)
#     Y_2 = f2(x + delta_t / 2 * X_1,
#              y + delta_t / 2 * Y_1,
#              v + delta_t / 2 * V_1,
#              angle + delta_t / 2 * P_1,
#              acc,
#              omega)
#     V_2 = f3(x + delta_t / 2 * X_1,
#              y + delta_t / 2 * Y_1,
#              v + delta_t / 2 * V_1,
#              angle + delta_t / 2 * P_1,
#              acc, omega)
#     P_2 = f4(x + delta_t / 2 * X_1,
#              y + delta_t / 2 * Y_1,
#              v + delta_t / 2 * V_1,
#              angle + delta_t / 2 * P_1,
#              acc, omega)
#     X_3 = f1(x + delta_t / 2 * X_2,
#              y + delta_t / 2 * Y_2,
#              v + delta_t / 2 * V_2,
#              angle + delta_t / 2 * P_2,
#              acc, omega)
#     Y_3 = f2(x + delta_t / 2 * X_2,
#              y + delta_t / 2 * Y_2,
#              v + delta_t / 2 * V_2,
#              angle + delta_t / 2 * P_2,
#              acc, omega)
#     V_3 = f3(x + delta_t / 2 * X_2,
#              y + delta_t / 2 * Y_2,
#              v + delta_t / 2 * V_2,
#              angle + delta_t / 2 * P_2,
#              acc, omega)
#     P_3 = f4(x + delta_t / 2 * X_2,
#              y + delta_t / 2 * Y_2,
#              v + delta_t / 2 * V_2,
#              angle + delta_t / 2 * P_2,
#              acc, omega)
#
#     X_4 = f1(x + delta_t * X_3,
#              y + delta_t * Y_3,
#              v + delta_t * V_3,
#              angle + delta_t * P_3,
#              acc, omega)
#     Y_4 = f2(x + delta_t * X_3,
#              y + delta_t * Y_3,
#              v + delta_t * V_3,
#              angle + delta_t * P_3,
#              acc, omega)
#     V_4 = f3(x + delta_t * X_3,
#              y + delta_t * Y_3,
#              v + delta_t * V_3,
#              angle + delta_t * P_3,
#              acc, omega)
#     P_4 = f4(x + delta_t * X_3,
#              y + delta_t * Y_3,
#              v + delta_t * V_3,
#              angle + delta_t * P_3,
#              acc, omega)
#     delta_x = (X_1 + 2 * X_2 + 2 * X_3 + X_4) * delta_t / 6
#     delta_y = (Y_1 + 2 * Y_2 + 2 * Y_3 + Y_4) * delta_t / 6
#     delta_v = (V_1 + 2 * V_2 + 2 * V_3 + V_4) * delta_t / 6
#     delta_angle = (P_1 + 2 * P_2 + 2 * P_3 + P_4) * delta_t / 6
#     return delta_x, delta_y, delta_v, delta_angle