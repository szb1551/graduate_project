import random
import math
import copy
import numpy as np
import pygame

# 采样时间
T = 0.1
# 机器人的最大线速度
MAX_V = 1.2
# 机器人的最大角速度(math.radians(300))
MAX_OMEGA = 0.8
# 时间步个数
N = 600
# 机器人的最大横坐标
MAX_X = 10
# 机器人的最大纵坐标
MAX_Y = 10
# 角度的归一化系数
MAX_THETA = math.pi
# 距离的归一化系数
MAX_D = math.sqrt(math.pow(MAX_X, 2) + math.pow(MAX_Y, 2))
# 死亡区的大小
MIN_D = 0.3


def reset():
    x_r = random.uniform(0.2 * MAX_X, 0.8 * MAX_X)
    y_r = 0.2 * MAX_Y
    theta_r = math.pi / 2
    x_b = random.uniform(0.2 * MAX_X, 0.8 * MAX_X)
    y_b = 0.8 * MAX_Y
    theta_b = -math.pi / 2
    s_r = [x_r, y_r, theta_r]
    s_b = [x_b, y_b, theta_b]
    return s_r, s_b


def observe(s_r, s_b):
    d = math.sqrt(math.pow(s_r[0] - s_b[0], 2) + math.pow(s_r[1] - s_b[1], 2))
    q_r = math.acos(((s_b[0] - s_r[0]) * math.cos(s_r[2]) + (s_b[1] - s_r[1]) * math.sin(s_r[2])) / d)
    q_b = math.acos(((s_r[0] - s_b[0]) * math.cos(s_b[2]) + (s_r[1] - s_b[1]) * math.sin(s_b[2])) / d)
    s = [q_r, q_b, d]
    return s


def normalize(s):
    s[0] = (2 * s[0] - MAX_THETA) / MAX_THETA
    s[1] = (2 * s[1] - MAX_THETA) / MAX_THETA
    s[2] = (2 * s[2] - MAX_D) / MAX_D


def step(s, a):
    v = (a[0] + 1) * MAX_V / 2
    omega = a[1] * MAX_OMEGA
    s_prime = copy.deepcopy(s)
    for i in range(int(T / 0.01)):
        s_prime[0] += v * math.cos(s_prime[2]) * 0.01
        s_prime[1] += v * math.sin(s_prime[2]) * 0.01
        s_prime[2] += omega * 0.01
    if s_prime[2] > math.pi:
        s_prime[2] -= 2 * math.pi
    if s_prime[2] < -math.pi:
        s_prime[2] += 2 * math.pi
    return s_prime


def evaluate(s):
    t_a = (s[1] - s[0]) / math.pi
    t_d = np.clip((MAX_D + MIN_D - 2 * s[2]) / (MAX_D - MIN_D), -1, 1)
    t = (t_a + 2 * t_d) / 3
    return t


def reward(s):
    r = 0
    if s[2] < MIN_D and s[0] + s[1] > math.radians(140):
        if s[0] < math.radians(30) < s[1]:
            r = 1
        if s[1] < math.radians(30) < s[0]:
            r = -1
    return r


def render():
    pass
