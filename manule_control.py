import os
import math
import random

import pygame
import matplotlib.pyplot as plt
import torch
import numpy as np
import my_ppo
import my_utils
from my_utils import boss_play
import my_ddpg
import ddpg_2v1
import re

color_library = ['r', 'g', 'b', 'purple', 'brown', 'black', 'gold']


def key_control(env):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                # Backspace to reset
                env.reset()
                # totalReward = 0
            if event.key == pygame.K_a:
                env.p1.a[1] += 0.1
                if env.p1.a[1] > 1:
                    env.p1.a[1] = 1
            if event.key == pygame.K_d:
                env.p1.a[1] -= 0.1
                if env.p1.a[1] < -1:
                    env.p1.a[1] = -1
            if event.key == pygame.K_w:
                env.p1.a[0] += 0.1
                # print(env.p1.a)
                if env.p1.a[0] > 1:
                    env.p1.a[0] = 1
            if event.key == pygame.K_s:
                env.p1.a[0] -= 0.1
                # print(env.p1.a)
                if env.p1.a[0] < -1:
                    env.p1.a[0] = -1


def key_control2(env):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                # Backspace to reset
                env.reset()
                # totalReward = 0
            if event.key == pygame.K_a:
                env.p1.move_turn(0.1)
            if event.key == pygame.K_d:
                env.p1.move_turn(-0.1)
            if event.key == pygame.K_w:
                env.p1.move_forward(0.1)
            if event.key == pygame.K_s:
                env.p1.move_forward(-0.1)


def manual_control(actor=None, all_action=[], control_way=0):
    env = boss_play()
    env.reset()
    done = False
    FPS = 60
    play_T = -1  # ????????????
    my_list = []
    np_list = []
    bo_list = []
    person_list = []
    clock = pygame.time.Clock()
    env.p2.s_train = env.p2.observe(env.bo.s)
    while not done:
        clock.tick(FPS)
        if control_way:
            key_control(env)
        else:
            key_control2(env)
        env.p2.normalize(env.p2.s_train)
        if actor != None:
            env.p2.a = [actor(torch.Tensor(env.p2.s_train).to(my_ddpg.device))[0].item(),
                        actor(torch.Tensor(env.p2.s_train).to(
                            my_ddpg.device))[1].item()]
        else:
            env.p2.a = [-1, 0]
        env.p2.s = env.p2.step(env.p2.a)
        env.p1.s = env.p1.step(env.p1.a)
        env.p1.s_train = env.p1.observe(env.bo.s)
        env.p2.s_train = env.p2.observe(env.bo.s)
        # print(env.p1.a)
        # print(person_list)
        # sample_person(person_list, env.p1.s_train, env.p1.a)  # ????????????s???a??????
        env.bo.a = env.bo.select_action(env.p1.s)
        env.bo.s = env.bo.step(env.bo.a)
        env.p1.reward_test()
        env.p2.reward_test()
        play_T += 1
        if play_T > FPS / 2:
            my_list.append(env.p1.s)
            bo_list.append(env.bo.s)
            np_list.append(env.p2.s)
            a = [env.p1.a[0], env.p1.a[1]]
            person_list.append(a)
            play_T = 0

        env.render()
        # print('s', env.p1.s)
        # print('s_train', env.p1.s_train)
        # print(env.p1.r)
        if env.p1.r + env.p2.r != 0:
            my_list.append(env.p1.s)
            bo_list.append(env.bo.s)
            np_list.append(env.p2.s)
            person_list.append(env.p1.a)
            all_action.extend(person_list)
            draw_action(person_list)
            return my_list, bo_list, np_list, env.p1.r + env.p2.r
            # break


def manual_control3(actor=None, control_way=0, num=1):
    env = boss_play()
    env.reset()
    done = False
    FPS = 60
    play_T = -1  # ????????????
    my_list = []
    np_list = []
    bo_list = []
    clock = pygame.time.Clock()
    env.p2.s_train = env.p2.observe(env.bo.s)
    while not done:
        clock.tick(FPS)
        if control_way:
            key_control(env)
        else:
            key_control2(env)
        env.p2.normalize(env.p2.s_train)
        if actor != None:
            if num == 1:
                env.p2.a = [actor(torch.Tensor(env.p2.s_train).to(my_ddpg.device))[0].item(),
                            actor(torch.Tensor(env.p2.s_train).to(
                                my_ddpg.device))[1].item()]
            else:
                env.p2.a = [actor(torch.Tensor(env.p2.s_train + env.p1.s_train).to(my_ddpg.device))[0].item(),
                            actor(torch.Tensor(env.p2.s_train + env.p1.s_train).to(
                                my_ddpg.device))[1].item()]
        else:
            env.p2.a = [-1, 0]
        env.p2.s = env.p2.step(env.p2.a)
        env.p1.s = env.p1.step(env.p1.a)
        env.p1.s_train = env.p1.observe(env.bo.s)
        env.p2.s_train = env.p2.observe(env.bo.s)
        # print(env.p1.a)
        # print(person_list)
        # sample_person(person_list, env.p1.s_train, env.p1.a)  # ????????????s???a??????
        env.bo.a = env.bo.select_action(env.p1.s)
        env.bo.s = env.bo.step(env.bo.a)
        env.p1.reward_test()
        env.p2.reward_test()
        play_T += 1
        if play_T > FPS / 2:
            my_list.append(env.p1.s)
            bo_list.append(env.bo.s)
            np_list.append(env.p2.s)
            play_T = 0

        env.render()
        # print('s', env.p1.s)
        # print('s_train', env.p1.s_train)
        # print(env.p1.r)
        if env.p1.r + env.p2.r != 0:
            my_list.append(env.p1.s)
            bo_list.append(env.bo.s)
            np_list.append(env.p2.s)
            return my_list, bo_list, np_list, env.p1.r + env.p2.r


def load_np(file="strategy.npy"):
    return np.load(file)


def save_np(file="style_times", arr=np.zeros(4)):
    np.save(file=file, arr=arr)


def record_style_times(style_times, index):
    if index in range(3):
        style_times[0] += 1
    elif index in range(3, 6):
        style_times[1] += 1
    elif index in range(6, 9):
        style_times[2] += 1
    else:
        style_times[3] += 1


def manual_control2(actor=None, control_way=0, threat_AI=True, record=False, style_times=np.zeros(4)):
    ac = choose_action(name="actor_pi")  # ?????????actor??????
    ac2 = choose_actor(name="actor_pi")  # ?????????????????????
    strategy = load_np()  # ?????????
    actor_on = ddpg_2v1.Actor(s_dim=3, a_dim=2).to(my_ddpg.device)  # ????????????
    actor_ti = ddpg_2v1.Actor_team(s_dim=3, a_dim=2).to(my_ddpg.device)  # ???????????????????????????
    actor_ppo = my_ppo.Pi_net()  # PPO??????
    actor2_on = ddpg_2v1.Actor(s_dim=3, a_dim=2).to(my_ddpg.device)
    actor2_ti = ddpg_2v1.Actor_team(s_dim=3, a_dim=2).to(my_ddpg.device)
    First = False  # ??????actor?????????
    Second = False
    actor_on.eval()
    actor_ti.eval()
    actor_ppo.eval()
    actor2_on.eval()
    actor2_ti.eval()
    all_symbol = []  # ??????actor??????state?????????
    symbol = []  # ???????????????
    sum = []  # ?????????????????????
    env = boss_play()
    env.reset()
    done = False
    FPS = 120
    play_T = -1  # ????????????
    choose_T = 0  # ????????????,????????????????????????
    my_list = []  # ????????????????????????
    np_list = []  # ???????????????????????????
    bo_list = []  # ???????????????????????????
    person_list = []  # ??????????????????
    clock = pygame.time.Clock()
    env.p2.s_train = env.p2.observe(env.bo.s)
    symbol_first = True  # ?????????????????????
    while not done:
        clock.tick(FPS)
        env.p1.s_train = env.p1.observe(env.bo.s)
        if control_way == 0:
            key_control2(env)
        else:
            key_control(env)

        person_list.append([env.p1.a[0], env.p1.a[1]])
        begin = 0
        for pi in ac:
            # ???????????????????????????
            if symbol_first:
                all_symbol.append([])
            try:
                actor_person = actor_on
                actor_person.load_state_dict(torch.load(pi))
                a = np.array([actor_person(torch.Tensor(env.p1.s_train).to(my_ddpg.device))[0].item(),
                              actor_person(torch.Tensor(env.p1.s_train).to(my_ddpg.device))[1].item()])
                all_symbol[begin].append(a)
            except RuntimeError:
                # print('????????????')
                try:
                    actor_person = actor_ti
                    actor_person.load_state_dict(torch.load(pi))
                    a = np.array(
                        [actor_person(torch.Tensor(env.p1.s_train + env.p2.s_train).to(my_ddpg.device))[0].item(),
                         actor_person(torch.Tensor(env.p1.s_train + env.p2.s_train).to(my_ddpg.device))[1].item()])
                    all_symbol[begin].append(a)
                except RuntimeError:
                    actor_person = actor_ppo
                    actor_person.load_state_dict(torch.load(pi))
                    a = ddpg_2v1.select_action(env.p1.s_train, actor=actor_person, method='PPO')
                    all_symbol[begin].append(a)
            begin += 1
        env.p2.normalize(env.p2.s_train)
        if symbol_first:  # ???????????????????????????????????????????????????
            symbol_first = False
        if choose_T == 29:
            choose_T = 0
            min_temp = 10000  # ????????????????????????
            sum.clear()
            # print(person_list)
            for sy in all_symbol:
                # print(sy)
                sum_sy = similarity(person_list, sy)
                if sum_sy < min_temp:
                    min_temp = sum_sy
                sum.append(sum_sy)
                sy.clear()
            print(sum)
            index = sum.index(min_temp)
            record_style_times(style_times=style_times, index=index)
            print(ac[index])
            actor2_pos = ac2[np.argmax(strategy[index])]
            print(actor2_pos)
            person_list.clear()
            try:
                actor = actor2_on
                actor.load_state_dict(torch.load(actor2_pos))
                First = True
                Second = False
            except RuntimeError:
                actor = actor2_ti
                actor.load_state_dict(torch.load(torch.load(actor2_pos)))
                First = False
                Second = True
        if actor != None:
            if First:
                env.p2.a = [actor(torch.Tensor(env.p2.s_train).to(my_ddpg.device))[0].item(),
                            actor(torch.Tensor(env.p2.s_train).to(
                                my_ddpg.device))[1].item()]
            if Second:
                env.p2.a = [actor(torch.Tensor(env.p2.s_train + env.p1.s_train).to(my_ddpg.device))[0].item(),
                            actor(torch.Tensor(env.p2.s_train + env.p1.s_train).to(
                                my_ddpg.device))[1].item()]
        else:
            if threat_AI:
                env.p2.a = env.p2.select_action(env.bo.s)
            else:
                env.p2.a = [-1, 0]
        env.p2.s = env.p2.step(env.p2.a)
        env.p1.s = env.p1.step(env.p1.a)
        env.p1.s_train = env.p1.observe(env.bo.s)
        env.p2.s_train = env.p2.observe(env.bo.s)
        # sample_person(person_list, env.p1.s_train, env.p1.a)  # ????????????s???a??????
        env.bo.a = env.bo.select_action(env.p1.s)
        env.bo.s = env.bo.step(env.bo.a)
        env.p1.reward_test()
        env.p2.reward_test()
        play_T += 1
        choose_T += 1
        # print(choose_T)
        if play_T > FPS / 2:
            my_list.append(env.p1.s)
            bo_list.append(env.bo.s)
            np_list.append(env.p2.s)
            play_T = 0

        env.render()
        if env.p1.r + env.p2.r != 0:
            my_list.append(env.p1.s)
            bo_list.append(env.bo.s)
            np_list.append(env.p2.s)
            if record:
                print("loading...")
                print(style_times)
                save_np(arr=style_times)
                return style_times
            else:
                return my_list, bo_list, np_list, env.p1.r + env.p2.r


def draw_action(action_list, c=None):  # ??????v???w?????????
    v_list = []
    w_list = []
    for ac in action_list:
        v_list.append(ac[0])
        w_list.append(ac[1])
    plt.scatter(w_list, v_list, c=c)
    plt.xlim(-1, 1)
    plt.ylim(-1.5, 1.5)
    plt.xlabel('w')
    plt.ylabel('v')
    plt.show()


def draw(list_0, list_1, list_2, r):
    x_0 = []
    y_0 = []
    x_1 = []
    y_1 = []
    x_npc = []
    y_npc = []
    for i in list_0:
        x_0.append(i[0])
        y_0.append(i[1])
    for j in list_1:
        x_1.append(j[0])
        y_1.append(j[1])
    for k in list_2:
        x_npc.append(k[0])
        y_npc.append(k[1])
    plt.figure(1)
    plt.scatter(x_0[0], y_0[0], c='r')
    plt.scatter(x_1[0], y_1[0], c='b')
    plt.scatter(x_npc[0], y_npc[0], c='g')
    plt.plot(x_0, y_0, color='r')
    plt.plot(x_1, y_1, 'b')
    plt.plot(x_npc, y_npc, 'g')
    if r > 0:
        plt.title('You wins')
    elif r < 0:
        plt.title('Blue wins')
    else:
        plt.title('Draw')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(0, my_utils.MAX_X)
    plt.ylim(0, my_utils.MAX_Y)
    plt.show()


def sample_person(person_list, s, a):  # ?????????????????????
    temp = [s, a]
    person_list.append(temp)


def choose_action(name='actor_all_pi'):  # ?????????????????????
    ac = []
    ore = re.compile(r'^pi|^actor\D')
    actors = os.listdir(name)  # ????????????????????????
    for actor_pis in actors:
        actors_name = name + '/' + actor_pis  # ??????????????????????????????
        # print(actors_name)
        actors_pi = os.listdir(actors_name)
        # print(actors_pi)
        for pi in actors_pi:
            if ore.search(pi):
                temp = actors_name + '/' + pi
                # print(temp)
                ac.append(temp)
    # print(ac)
    return ac


def choose_actor(name='actor_all_pi'):  # ?????????????????????
    ac = []
    ore = re.compile(r'^actor\d')
    actors = os.listdir(name)  # ????????????????????????
    for actor_pis in actors:
        actors_name = name + '/' + actor_pis  # ??????????????????????????????
        # print(actors_name)
        actors_pi = os.listdir(actors_name)
        # print(actors_pi)
        for pi in actors_pi:
            if ore.search(pi):
                temp = actors_name + '/' + pi
                # print(temp)
                ac.append(temp)
    return ac


def dis(pos1, pos2):  # ????????????????????????
    return math.sqrt(math.pow(pos1[0] - pos2[0], 2) + math.pow(pos1[1] - pos2[1], 2))


def similarity(list1, list2):  # ????????????????????????
    sum = 0
    for i in range(len(list1)):
        # print(i)a
        sum += dis(list1[i], list2[i])
    return round(sum, 2)


def draw_pie_chart(file="style_times.npy"):
    styles = load_np(file=file)
    sum_styles = 0
    for time in styles:
        sum_styles += time
    plt.figure(figsize=(6, 6))  # ???????????????????????????
    label = 'combat', 'delay', 'die', 'run'  # ???????????????
    sizes = []  # ???????????????
    for time in styles:
        sizes.append(time / sum_styles)
    color = 'g', 'r', 'b', 'y', 'c'  # ???????????????
    explode = (0, 0, 0, 0)  # ????????????????????????
    patches, text1, text2 = plt.pie(sizes,
                                    colors=color,
                                    explode=explode,
                                    labels=label,
                                    shadow=False,  # ???????????????
                                    autopct="%1.1f%%",  # ???????????????????????????
                                    startangle=90,  # ?????????????????????
                                    pctdistance=0.6)  # ?????????????????????????????????
    # patches?????????????????????text1?????????label????????????text2?????????????????????

    plt.axis('equal')  # ??????????????????
    plt.legend()
    plt.show()


def num_ac2(ac2, index):  # ?????????????????????????????????????????????
    ore = re.compile(r"team")
    if ore.search(ac2[index]):
        return 2
    return 1


def draw_pic(list1, list2):  # ???????????????
    x_0 = []
    for i in range(10):
        x_0.append(i)
    plt.scatter(x_0, list1)  # ?????????????????????
    plt.scatter(x_0, list2)  # ??????????????????
    plt.xlabel('Trails')
    plt.ylabel('Performance')
    l1, = plt.plot(x_0, list1, 'r')
    l2, = plt.plot(x_0, list2, 'b')
    plt.legend(handles=[l1, l2], labels=["Adaptive", "Random"], loc="best")
    plt.show()


if __name__ == '__main__':
    for i in range(100):
        manual_control2()
