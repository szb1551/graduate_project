import copy

import my_ddpg
import torch

import my_ppo
import my_utils
import matplotlib.pyplot as plt
from my_utils import boss_play
import pygame
import math
import ddpg_2v1


def dd_test(actor):
    # s_r, s_b = utils.reset()
    p1 = my_utils.players()
    bo = my_utils.boss()
    p2_npc = my_utils.players()
    p1.reset()
    bo.reset()
    p2_npc.reset()
    p1.s_train = p1.observe(bo.s)
    list_0 = [[p1.s[0], p1.s[1]]]  # 自己的位置
    list_1 = [[bo.s[0], bo.s[1]]]  # 敌方的位置
    list_2 = [[p2_npc.s[0], p2_npc.s[1]]]
    for i in range(my_utils.N):
        # utils.normalize(s)
        p1.normalize(p1.s_train)
        a_r = [actor(torch.Tensor(p1.s_train).to(my_ddpg.device))[0].item(), actor(torch.Tensor(p1.s_train).to(
            my_ddpg.device))[1].item()]
        a_npc = p2_npc.select_action(bo.s)
        a_b = bo.select_action(p1.s)
        p1.s = p1.step(a_r)
        bo.s = bo.step(a_b)
        # print(a_npc)
        p2_npc.s = p2_npc.step(a_npc)
        p1.s_train = p1.observe(bo.s)
        p2_npc.s_train = p2_npc.observe(bo.s)
        list_0.append([p1.s[0], p1.s[1]])
        list_1.append([bo.s[0], bo.s[1]])
        list_2.append([p2_npc.s[0], p2_npc.s[1]])
        if bo.s[0] < 0 or bo.s[0] > my_utils.MAX_X or bo.s[1] < 0 or bo.s[1] > my_utils.MAX_Y:
            p1.r = 1
        elif p1.s[0] < 0 or p1.s[0] > my_utils.MAX_X or p1.s[1] < 0 or p1.s[1] > my_utils.MAX_Y:
            p1.r = -1
        else:
            # r = utils.reward(s)
            p1.reward_test()
            p2_npc.reward_test()
            print(p2_npc.r)
            if p2_npc.r:
                p1.r += p2_npc.r
            # print(p2_npc.r)
        # print(p1.r)
        # print(p1.s)
        # print(bo.s)
        # print(a_r)
        # print(a_b)
        if p1.r != 0:
            break
    return list_0, list_1, list_2, p1.r


def dd_test2(actor, actor2):
    # s_r, s_b = utils.reset()
    p1 = my_utils.players()
    bo = my_utils.boss()
    p2 = my_utils.players()
    p1.reset()
    bo.reset()
    p2.reset(range_x_1=0.4, range_x_2=0.8)
    p1.s_train = p1.observe(bo.s)
    p2.s_train = p2.observe(bo.s)
    list_0 = [[p1.s[0], p1.s[1]]]  # 自己的位置
    list_1 = [[bo.s[0], bo.s[1]]]  # 敌方的位置
    list_2 = [[p2.s[0], p2.s[1]]]
    for i in range(my_utils.N):
        # utils.normalize(s)
        p1.normalize(p1.s_train)
        p2.normalize(p2.s_train)
        a_r = [actor(torch.Tensor(p1.s_train).to(my_ddpg.device))[0].item(), actor(torch.Tensor(p1.s_train).to(
            my_ddpg.device))[1].item()]
        a_npc = [actor2(torch.Tensor(p2.s_train).to(my_ddpg.device))[0].item(), actor2(torch.Tensor(p2.s_train).to(
            my_ddpg.device))[1].item()]
        a_b = bo.select_action(p1.s)
        p1.s = p1.step(a_r)
        bo.s = bo.step(a_b)
        # print(a_npc)
        p2.s = p2.step(a_npc)
        p1.s_train = p1.observe(bo.s)
        p2.s_train = p2.observe(bo.s)
        list_0.append([p1.s[0], p1.s[1]])
        list_1.append([bo.s[0], bo.s[1]])
        list_2.append([p2.s[0], p2.s[1]])
        if bo.s[0] < 0 or bo.s[0] > my_utils.MAX_X or bo.s[1] < 0 or bo.s[1] > my_utils.MAX_Y:
            p1.r = 1
        elif p1.s[0] < 0 or p1.s[0] > my_utils.MAX_X or p1.s[1] < 0 or p1.s[1] > my_utils.MAX_Y:
            p1.r = -1
        else:
            # r = utils.reward(s)
            p1.reward_test()
            p2.reward_test()
            # print(p2.r)
            if p2.r:
                p1.r += p2.r
            # print(p2_npc.r)
        # print(p1.r)
        # print(p1.s)
        # print(bo.s)
        # print(a_r)
        # print(a_b)
        if p1.r != 0:
            break
    return list_0, list_1, list_2, p1.r


def ddpg_render(actor):
    env = boss_play()
    env.reset()
    done = False
    FPS = 60
    play_T = -1  # 采集时间
    env.p1.s_train = env.p1.observe(env.bo.s)
    my_list = []
    np_list = []
    bo_list = []
    clock = pygame.time.Clock()
    env.p1.s_train = env.p1.observe(env.bo.s)
    while not done:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return my_list, bo_list, np_list, env.p1.r + env.p2.r
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # Backspace to reset
                    env.reset()
                    # totalReward = 0

        env.p1.normalize(env.p1.s_train)
        env.p1.a = [actor(torch.Tensor(env.p1.s_train).to(my_ddpg.device))[0].item(),
                    actor(torch.Tensor(env.p1.s_train).to(
                        my_ddpg.device))[1].item()]
        # print(env.p1.a)
        env.p2.a = env.p2.select_action(env.bo.s)
        env.bo.a = env.bo.select_action(env.p1.s)
        env.p1.s = env.p1.step(env.p1.a)
        env.p2.s = env.p2.step(env.p2.a)
        env.bo.s = env.bo.step(env.bo.a)
        env.p1.s_train = env.p1.observe(env.bo.s)
        env.p2.s_train = env.p2.observe(env.bo.s)
        env.p1.reward_test()
        env.p2.reward_test()
        # print(math.degrees(env.p2.s_train[0]), math.degrees(env.p2.s_train[1]))
        play_T += 1
        if play_T > FPS / 2:
            # print(env.p2.s_train)
            # print(env.p2.s)
            # print(math.degrees(env.p2.s_train[0]),math.degrees(env.p2.s_train[1]))
            # print(env.p1.s)
            my_list.append(env.p1.s)
            bo_list.append(env.bo.s)
            np_list.append(env.p2.s)
            play_T = 0

        env.render()
        if env.p1.r + env.p2.r != 0:
            my_list.append(env.p1.s)
            bo_list.append(env.bo.s)
            np_list.append(env.p2.s)
            return my_list, bo_list, np_list, env.p1.r + env.p2.r


def ddpg_render2(actor=None, actor2=None, way=0, num=1):
    env = boss_play()
    env.reset()
    done = False
    FPS = 60
    play_T = -1  # 采集时间
    my_list = []
    np_list = []
    bo_list = []
    clock = pygame.time.Clock()
    env.p1.s_train = env.p1.observe(env.bo.s)
    env.p2.s_train = env.p2.observe(env.bo.s)
    times = 0  # 1分钟内解决
    while not done and times < 60:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return my_list, bo_list, np_list, env.p1.r + env.p2.r
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # Backspace to reset
                    env.reset()
                    # totalReward = 0

        env.p1.normalize(env.p1.s_train)
        env.p2.normalize(env.p2.s_train)
        if actor:
            if num == 1:
                env.p1.a = [actor(torch.Tensor(env.p1.s_train).to(my_ddpg.device))[0].item(),
                            actor(torch.Tensor(env.p1.s_train).to(
                                my_ddpg.device))[1].item()]
            else:
                env.p1.a = [actor(torch.Tensor(env.p1.s_train + env.p2.s_train).to(my_ddpg.device))[0].item(),
                            actor(torch.Tensor(env.p1.s_train + env.p2.s_train).to(
                                my_ddpg.device))[1].item()]
        else:
            env.p1.a = [-1, 0]
        # print('p1',env.p1.a)
        if actor2:
            if num == 1:
                env.p2.a = [actor2(torch.Tensor(env.p2.s_train).to(my_ddpg.device))[0].item(),
                            actor2(torch.Tensor(env.p2.s_train).to(
                                my_ddpg.device))[1].item()]
            else:
                env.p2.a = [actor2(torch.Tensor(env.p2.s_train + env.p1.s_train).to(my_ddpg.device))[0].item(),
                            actor2(torch.Tensor(env.p2.s_train + env.p1.s_train).to(
                                my_ddpg.device))[1].item()]
        else:
            env.p2.a = [-1, 0]
        # print(env.p2.a)
        if way:
            env.bo.a = env.bo.select_action(env.p2.s)
        else:
            env.bo.a = env.bo.select_action(env.p1.s)
        env.p1.s = env.p1.step(env.p1.a)
        env.p2.s = env.p2.step(env.p2.a)
        env.bo.s = env.bo.step(env.bo.a)
        env.p1.s_train = env.p1.observe(env.bo.s)
        env.p2.s_train = env.p2.observe(env.bo.s)
        env.p1.reward_test()
        env.p2.reward_test()
        # print(math.degrees(env.p2.s_train[0]), math.degrees(env.p2.s_train[1]))
        play_T += 1
        if play_T > FPS / 2:
            # print(env.p2.s_train)
            # print(env.p2.s)
            # print(math.degrees(env.p2.s_train[0]),math.degrees(env.p2.s_train[1]))
            # print(env.p1.s)
            times += 0.5
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
    return my_list, bo_list, np_list, env.p1.r + env.p2.r


def test_render(actor=None, method1='DDPG', actor2=None, method2='DDPG', way=0, num1=1, num2=1, run=False):
    env = boss_play()
    env.reset()
    done = False
    FPS = 120
    play_T = -1  # 采集时间
    my_list = []
    np_list = []
    bo_list = []
    clock = pygame.time.Clock()
    env.p1.s_train = env.p1.observe(env.bo.s)
    env.p2.s_train = env.p2.observe(env.bo.s)
    times = 0  # 1分钟内解决
    while not done and times < 60:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return my_list, bo_list, np_list, env.p1.r + env.p2.r
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # Backspace to reset
                    env.reset()
                    # totalReward = 0

        env.p1.normalize(env.p1.s_train)
        env.p2.normalize(env.p2.s_train)
        if actor:
            if num1 == 1:
                if method1 == 'DDPG':
                    env.p1.a = [actor(torch.Tensor(env.p1.s_train).to(my_ddpg.device))[0].item(),
                                actor(torch.Tensor(env.p1.s_train).to(
                                    my_ddpg.device))[1].item()]
                elif method1 == 'PPO':
                    env.p1.a = ddpg_2v1.select_action(env.p1.s_train, actor=actor, method=method1)

            else:
                if method1 == 'DDPG':
                    env.p1.a = [actor(torch.Tensor(env.p1.s_train + env.p2.s_train).to(my_ddpg.device))[0].item(),
                                actor(torch.Tensor(env.p1.s_train + env.p2.s_train).to(
                                    my_ddpg.device))[1].item()]
                elif method1 == 'PPO':
                    env.p1.a = ddpg_2v1.select_action(env.p1.s_train, actor=actor, method=method1)
        else:
            env.p1.a = [-1, 0]
        # print('p1',env.p1.a)
        if actor2:
            if num2 == 1:
                if method2 == 'DDPG':
                    env.p2.a = [actor2(torch.Tensor(env.p2.s_train).to(my_ddpg.device))[0].item(),
                                actor2(torch.Tensor(env.p2.s_train).to(
                                    my_ddpg.device))[1].item()]
                elif method2 == 'PPO':
                    env.p2.a = ddpg_2v1.select_action(env.p2.s_train, actor=actor2, method=method2)
            else:
                # env.p2.a = [actor2(torch.Tensor(env.p2.s_train + env.p1.s_train).to(my_ddpg.device))[0].item(),
                #             actor2(torch.Tensor(env.p2.s_train + env.p1.s_train).to(
                #                 my_ddpg.device))[1].item()]
                env.p2.a = ddpg_2v1.select_action(env.p2.s_train + env.p1.s_train, actor=actor2, method=method2)
        else:
            env.p2.a = [-1, 0]
        # print(env.p2.a)
        if way:
            env.bo.a = env.bo.select_action(env.p2.s)
        else:
            env.bo.a = env.bo.select_action(env.p1.s)
        env.p1.s = env.p1.step(env.p1.a)
        env.p2.s = env.p2.step(env.p2.a)
        env.bo.s = env.bo.step(env.bo.a)
        env.p1.s_train = env.p1.observe(env.bo.s)
        env.p2.s_train = env.p2.observe(env.bo.s)
        env.p1.reward_test()
        env.p2.reward_test()
        # print(math.degrees(env.p2.s_train[0]), math.degrees(env.p2.s_train[1]))
        play_T += 1
        if play_T > 30:
            # print(env.p2.s_train)
            # print(env.p2.s)
            # print(math.degrees(env.p2.s_train[0]),math.degrees(env.p2.s_train[1]))
            # print(env.p1.s)
            times += 1
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
    return my_list, bo_list, np_list, env.p1.r + env.p2.r


def test_no_render(actor=None, method1='DDPG', actor2=None, method2='DDPG', way=0, num1=1, num2=1):
    env = my_utils.train_play()
    env.reset()
    done = False
    play_T = -1  # 采集时间
    my_list = []
    np_list = []
    bo_list = []
    env.p1.s_train = env.p1.observe(env.bo.s)
    env.p2.s_train = env.p2.observe(env.bo.s)
    times = 0  # 1分钟内解决
    while not done and times < 60:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return my_list, bo_list, np_list, env.p1.r + env.p2.r
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # Backspace to reset
                    env.reset()
                    # totalReward = 0

        env.p1.normalize(env.p1.s_train)
        env.p2.normalize(env.p2.s_train)
        if actor:
            if num1 == 1:
                if method1 == 'DDPG':
                    env.p1.a = [actor(torch.Tensor(env.p1.s_train).to(my_ddpg.device))[0].item(),
                                actor(torch.Tensor(env.p1.s_train).to(
                                    my_ddpg.device))[1].item()]
                elif method1 == 'PPO':
                    env.p1.a = ddpg_2v1.select_action(env.p1.s_train, actor=actor, method=method1)

            else:
                if method1 == 'DDPG':
                    env.p1.a = [actor(torch.Tensor(env.p1.s_train + env.p2.s_train).to(my_ddpg.device))[0].item(),
                                actor(torch.Tensor(env.p1.s_train + env.p2.s_train).to(
                                    my_ddpg.device))[1].item()]
                elif method1 == 'PPO':
                    env.p1.a = ddpg_2v1.select_action(env.p1.s_train, actor=actor, method=method1)
        else:
            env.p1.a = [-1, 0]
        # print('p1',env.p1.a)
        if actor2:
            if num2 == 1:
                if method2 == 'DDPG':
                    env.p2.a = [actor2(torch.Tensor(env.p2.s_train).to(my_ddpg.device))[0].item(),
                                actor2(torch.Tensor(env.p2.s_train).to(
                                    my_ddpg.device))[1].item()]
                elif method2 == 'PPO':
                    env.p2.a = ddpg_2v1.select_action(env.p2.s_train, actor=actor2, method=method2)
            else:
                # env.p2.a = [actor2(torch.Tensor(env.p2.s_train + env.p1.s_train).to(my_ddpg.device))[0].item(),
                #             actor2(torch.Tensor(env.p2.s_train + env.p1.s_train).to(
                #                 my_ddpg.device))[1].item()]
                env.p2.a = ddpg_2v1.select_action(env.p2.s_train + env.p1.s_train, actor=actor2, method=method2)
        else:
            env.p2.a = [-1, 0]
        # print(env.p2.a)
        if way:
            env.bo.a = env.bo.select_action(env.p2.s)
        else:
            env.bo.a = env.bo.select_action(env.p1.s)
        env.p1.s = env.p1.step(env.p1.a)
        env.p2.s = env.p2.step(env.p2.a)
        env.bo.s = env.bo.step(env.bo.a)
        env.p1.s_train = env.p1.observe(env.bo.s)
        env.p2.s_train = env.p2.observe(env.bo.s)
        env.p1.reward_test()
        env.p2.reward_test()
        # print(math.degrees(env.p2.s_train[0]), math.degrees(env.p2.s_train[1]))
        play_T += 1
        if play_T > 30:
            # print(env.p2.s_train)
            # print(env.p2.s)
            # print(math.degrees(env.p2.s_train[0]),math.degrees(env.p2.s_train[1]))
            # print(env.p1.s)
            times += 1
            my_list.append(env.p1.s)
            bo_list.append(env.bo.s)
            np_list.append(env.p2.s)
            play_T = 0

        if env.p1.r + env.p2.r != 0:
            my_list.append(env.p1.s)
            bo_list.append(env.bo.s)
            np_list.append(env.p2.s)
            return my_list, bo_list, np_list, env.p1.r + env.p2.r
    return my_list, bo_list, np_list, env.p1.r + env.p2.r


def draw(list_0, list_1, list_2, r):  # 画三方的角力图
    x_0 = []
    y_0 = []
    x_1 = []
    y_1 = []
    x_npc = []
    y_npc = []
    for i in list_0:
        print(i)
        x_0.append(i[0])
        y_0.append(i[1])
    for j in list_1:
        x_1.append(j[0])
        y_1.append(j[1])
    for k in list_2:
        x_npc.append(k[0])
        y_npc.append(k[1])
    plt.figure(1)
    plt.scatter(x_0[0], y_0[0])
    plt.scatter(x_1[0], y_1[0])
    plt.scatter(x_npc[0], y_npc[0])
    plt.plot(x_0, y_0, 'r')
    plt.plot(x_1, y_1, 'b')
    plt.plot(x_npc, y_npc, 'g')
    if r == 1:
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


def draw_one(list_0, list_1,r,name="combat"):  # 只画一方的图
    x_0 = []
    y_0 = []
    x_1 = []
    y_1 = []
    for i in list_0:
        x_0.append(i[0])
        y_0.append(i[1])
    for j in list_1:
        x_1.append(j[0])
        y_1.append(j[1])
    plt.scatter(x_0[0], y_0[0])
    plt.scatter(x_1[0], y_1[0])
    l1,=plt.plot(x_0, y_0, 'r')
    l2,=plt.plot(x_1, y_1, 'b')
    plt.xlabel('x')
    plt.ylabel('y')
    if r>0:
        plt.title("Wins")
    elif r<0:
        plt.title("Loss")
    else:
        plt.title("Tie")
    plt.xlim(0, my_utils.MAX_X)
    plt.ylim(0, my_utils.MAX_Y)
    plt.legend(handles=[l1, l2],labels=[name,"threat"],loc="best")
    plt.show()


if __name__ == "__main__":
    sum_r = 0

    # actor = ddpg_2v1.Actor_team(s_dim=3, a_dim=2).to(my_ddpg.device)
    actor = my_ddpg.Actor(s_dim=3, a_dim=2).to(my_ddpg.device)
    # actor = my_ppo.Pi_net()
    # actor2 = my_ddpg.Actor(s_dim=3, a_dim=2).to(my_ddpg.device)
    actor2 = ddpg_2v1.Actor_team(s_dim=3,a_dim=2).to(my_ddpg.device)
    # actor2 = copy.deepcopy(actor)
    actor.load_state_dict(torch.load('actor_all_pi/attacter/actor_ddpg.pth'))
    # actor.load_state_dict(torch.load('pi.pth'))
    # actor2.load_state_dict(torch.load('actor_all_pi/actor_nomove/actor2_2000.pth'))
    # actor.load_state_dict(torch.load('actor.pth'))
    # actor.load_state_dict(torch.load('actor_pi/逃跑/pi2.pth'))
    # actor.load_state_dict(torch.load('actor.pth'))
    # actor2.load_state_dict(torch.load('actor2.pth'))
    # actor.load_state_dict(torch.load('actor_team.pth'))0
    # actor2.load_state_dict(torch.load('actor_all_pi/runner/actor2_with_run.pth'))
    # actor2.load_state_dict(torch.load('actor_pi/送死/actor2_2.pth'))
    actor2.load_state_dict(torch.load('actor2_team.pth'))
    actor.eval()
    actor2.eval()
    for i in range(100):
        # list_0, list_1, list_2, r = dd_test(actor)
        # list_0, list_1, list_2, r = dd_test2(actor, actor2)
        # list_0, list_1, list_2, r = ddpg_render(actor)
        # list_0, list_1, list_2, r = ddpg_render2(actor=actor, actor2=actor2, way=0, num=1)
        list_0, list_1, list_2, r = test_render(actor=actor, method1='DDPG', actor2=actor2, way=0, num1=1,num2=2)
        # list_0, list_1, list_2, r = test_no_render(actor=actor, method1='DDPG', actor2=None, way=0, num1=2, num2=1)
        if r == 1:
            sum_r += 1
        # print(len(list_0), len(list_1), len(list_2))
        draw(list_0, list_1, list_2, r)
        # draw_one(list_0, list_1,r,name="run")
    print(sum_r)
    # draw(list_0, list_1, r)
