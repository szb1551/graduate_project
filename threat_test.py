import torch
import random
import my_utils
import matplotlib.pyplot as plt
import pygame
import math


def draw(list_0, list_2, name=1):
    x_0 = []
    y_0 = []
    x_npc = []
    y_npc = []
    labels = []
    for i in list_0:
        x_0.append(i[0])
        y_0.append(i[1])
    for k in list_2:
        x_npc.append(k[0])
        y_npc.append(k[1])
    plt.figure(1)
    plt.scatter(x_0[0], y_0[0])
    plt.scatter(x_npc[0], y_npc[0])
    l1, = plt.plot(x_0, y_0, color='r')
    l2, = plt.plot(x_npc, y_npc, color='b')
    if name == 1:
        plt.title('straight_threat')
        labels = ['Straight', 'Threat']
    else:
        plt.title('random_threat')
        labels = ['Turn', 'Threat']
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(0, my_utils.MAX_X)
    plt.ylim(0, my_utils.MAX_Y)
    plt.legend(handles=[l1, l2], labels=labels, loc='best')
    plt.show()


def straight_threat():
    p_list = []
    b_list = []
    p1 = my_utils.players()
    p1.reset(theta=0)
    bo = my_utils.boss()
    bo.reset()
    for i in range(my_utils.N):
        p1.a = [0.2, 0]  # 直走
        bo.a = bo.select_action(p1.s)
        p1.s = p1.step(p1.a)
        # print(p1.s)
        bo.s = bo.step(bo.a)
        p_list.append(p1.s)
        b_list.append(bo.s)
        p1.s_train = p1.observe(bo.s)
        p1.reward_test()
        if p1.r != 0:
            p_list.append(p1.s)
            b_list.append(bo.s)
            draw(p_list, b_list, name=1)
            return
    print('未能结束')
    draw(p_list, b_list, name=1)


def turn_threat():
    p_list = []
    b_list = []
    p1 = my_utils.players()
    p1.reset(theta=0)
    bo = my_utils.boss()
    bo.reset()
    for i in range(my_utils.N):
        p1.a = [0.2, 0.1]  # 转向
        bo.a = bo.select_action(p1.s)
        p1.s = p1.step(p1.a)
        # print(p1.s)
        bo.s = bo.step(bo.a)
        p_list.append(p1.s)
        b_list.append(bo.s)
        p1.s_train = p1.observe(bo.s)
        p1.reward_test()
        if p1.r != 0:
            p_list.append(p1.s)
            b_list.append(bo.s)
            draw(p_list, b_list, name=2)
            return
    print('未能结束')
    draw(p_list, b_list, name=2)


if __name__ == "__main__":
    turn_threat()
