import random
import math
import copy
import numpy as np
import pygame
from resource.load import *
import time

# 采样时间
T = 0.1
# 机器人的最大线速度
MAX_V = 1.2
# 机器人的最大角速度(math.radians(300))
MAX_OMEGA = 0.8
# 时间步个数
N = 600
# 机器人的最大横坐标
# MAX_X = 10
MAX_X = 300
# 机器人的最大纵坐标
# MAX_Y = 10
MAX_Y = 300
# 角度的归一化系数
MAX_THETA = math.pi
# 距离的归一化系数
MAX_D = math.sqrt(math.pow(MAX_X, 2) + math.pow(MAX_Y, 2))
# 死亡区的大小(20)
MIN_D = 20


class players():  # 玩家，队友等设置
    def __init__(self):  # 参数设置
        self.a_space = ['v', 'omega']
        self.s_space = ['qr', 'qb', 'd']
        self.a = []  # 动作
        self.x = 0.0
        self.y = 0.0
        self.theta = math.pi / 2
        self.s = [self.x, self.y, self.theta]  # 绝对状态
        self.s_prime = []  # 绝对坐标的预测
        self.s_train = []  # observe处理后的相对状态
        self.s_train_prime = []  # 下一步相对状态的预测
        self.r = 0

    def reset(self, range_x_1=0.2, range_x_2=0.4, range_y=0.2,theta=math.pi/2):  # 初始化参数设置
        self.x = random.uniform(range_x_1 * MAX_X, range_x_2 * MAX_X)
        self.y = range_y * MAX_Y
        self.theta = theta
        self.a = [-1, 0]
        self.s = [self.x, self.y, self.theta]
        self.r = 0
        # return self.s

    def normalize(self, s_train):  # 均值化
        s_train[0] = (2 * s_train[0] - MAX_THETA) / MAX_THETA
        s_train[1] = (2 * s_train[1] - MAX_THETA) / MAX_THETA
        s_train[2] = (2 * s_train[2] - MAX_D) / MAX_D

    def observe(self, boss_s):  # 返回威胁指数
        d = math.sqrt(math.pow(self.s[0] - boss_s[0], 2) + math.pow(self.s[1] - boss_s[1], 2))+0.01
        q_r = math.acos(
            ((boss_s[0] - self.s[0]) * math.cos(self.s[2]) + (boss_s[1] - self.s[1]) * math.sin(self.s[2])) / d)
        q_b = math.acos(
            ((self.s[0] - boss_s[0]) * math.cos(boss_s[2]) + (self.s[1] - boss_s[1]) * math.sin(boss_s[2])) / d)
        return [q_r, q_b, d]

    def observe_prime(self, s_1):  # 预测威胁指数，用于npc的行动
        d = math.sqrt(math.pow(self.s_prime[0] - s_1[0], 2) + math.pow(self.s_prime[1] - s_1[1], 2))
        q_r = math.acos(
            ((s_1[0] - self.s_prime[0]) * math.cos(self.s_prime[2]) + (s_1[1] - self.s_prime[1]) * math.sin(
                self.s_prime[2])) / d)
        q_b = math.acos(
            ((self.s_prime[0] - s_1[0]) * math.cos(s_1[2]) + (self.s_prime[1] - s_1[1]) * math.sin(s_1[2])) / d)
        return [q_r, q_b, d]

    def step(self, a):  # 预测做出动作后的之后状态，预测的为绝对状态
        v = (a[0] + 1) * MAX_V / 2  # 线速度
        omega = a[1] * MAX_OMEGA  # 角速度
        s_prime = copy.deepcopy(self.s)  # 拷贝当前绝对坐标s
        for i in range(int(T / 0.01)):  # 分批
            # s_prime[0] += v * math.cos(s_prime[2]) * 0.01
            # s_prime[1] += v * math.sin(s_prime[2]) * 0.01
            s_prime[0] += v * math.cos(s_prime[2]) * 0.1
            s_prime[1] += v * math.sin(s_prime[2]) * 0.1
            s_prime[2] += omega * 0.01
        if s_prime[2] > math.pi:
            s_prime[2] -= 2 * math.pi
        if s_prime[2] < -math.pi:
            s_prime[2] += 2 * math.pi
        return s_prime

    def evalute(self, s):  # 评估威胁指数
        t_a = (s[1] - s[0]) / math.pi
        t_d = np.clip((MAX_D + MIN_D - 2 * s[2]) / (MAX_D - MIN_D), -1, 1)
        t = (t_a + 2 * t_d) / 3
        return t

    def evalute2(self, i):  # 生存时间
        return i / 600

    def evalute3(self, s):  # 威胁指数 作死版
        t_a = (s[1] - s[0]) / math.pi
        t_d = np.clip((MAX_D + MIN_D - 2 * s[2]) / (MAX_D - MIN_D), -1.5, 1.5)
        t = (t_a + 2 * t_d) / 3
        return -t

    def evalute4(self,s):  #
        # s_ = self.s # 当前的绝对状态
        if self.s[0] < 0+20 or self.s[0] > MAX_X-20 or self.s[1] < 0+20 or self.s[1] > MAX_Y-20:
            r = -0.05
            return r
        d = s[2]
        return d / 200

    def select_action(self, boss_s_prime):  # 若为AI队友，相对于对手的策略动作选择最大威胁的行为
        t_max = -1
        a = []
        for i in range(3):
            for j in range(3):
                a_b = [i - 1, j - 1]
                # s_b_prime = utils.step(boss_s, a_b)
                self.s_prime = self.step(a_b)  # 绝对
                s = self.observe_prime(boss_s_prime)  # 相对
                t = (MAX_D + MIN_D - 2 * s[2]) / (MAX_D - MIN_D)
                if t > t_max:
                    t_max = t
                    a = a_b
        return a

    def reward(self):  # 预测结束回报
        if self.s[0] < 0 or self.s[0] > MAX_X or self.s[1] < 0 or self.s[1] > MAX_Y:
            self.r = -10
        if self.s_train_prime[2] < MIN_D and self.s_train_prime[0] + self.s_train_prime[1] > math.radians(140):
            if self.s_train_prime[0] < math.radians(30) < self.s_train_prime[1]:
                self.r = 2
            if self.s_train_prime[1] < math.radians(30) < self.s_train_prime[0]:
                self.r = -2

    def reward2(self, r):
        if self.s[0] < 0 or self.s[0] > MAX_X or self.s[1] < 0 or self.s[1] > MAX_Y:
            self.r = -100
        if self.s_train_prime[2] > MIN_D and r != 0:
            self.r = 20  # 逃跑成功时的奖励
        elif self.s_train_prime[2] < MIN_D and self.s_train_prime[0] + self.s_train_prime[1] > math.radians(140):
            if self.s_train_prime[0] < math.radians(30) < self.s_train_prime[1]:
                self.r = 5
            if self.s_train_prime[1] < math.radians(30) < self.s_train_prime[0]:
                self.r = -2

    def reward3(self): # 成功送死
        if self.s[0] < 0 or self.s[0] > MAX_X or self.s[1] < 0 or self.s[1] > MAX_Y:
            self.r = 10
        elif self.s_train_prime[2] < MIN_D and self.s_train_prime[0] + self.s_train_prime[1] > math.radians(140):
            if self.s_train_prime[0] < math.radians(30) < self.s_train_prime[1]:
                self.r = -2
            if self.s_train_prime[1] < math.radians(30) < self.s_train_prime[0]:
                self.r = 10

    def reward_test(self):  # 测试运行时的回报
        if self.s[0] < 0 or self.s[0] > MAX_X or self.s[1] < 0 or self.s[1] > MAX_Y:
            self.r = -1
        if self.s_train[2] < MIN_D and self.s_train[0] + self.s_train[1] > math.radians(140):
            # math.radians()从度转换为弧度
            if self.s_train[0] < math.radians(30) < self.s_train[1]:
                self.r = 1
            if self.s_train[1] < math.radians(30) < self.s_train[0]:
                self.r = -5

    def move_forward(self, a_step):  # 玩家向前移动
        while True:
            self.a[0] += a_step
            if self.a[0] > 1:
                self.a[0] = 1
                break
            if self.a[0] < -1:
                self.a[0] = -1
                break

    def move_turn(self, w_step):  # 玩家转弯移动
        # self.a[1] += w_step
        while True:
            self.a[1] += w_step
            if self.a[1] > 0.5:
                self.a[1] = 0.5
                break
            if self.a[1] < -0.5:
                self.a[1] = -0.5
                break


class boss:  # 对手，boss参数待设置、默认普通对手
    def __init__(self):
        self.a_space = ['v', 'omega']
        # self.a = []
        self.x = 0.0
        self.y = 0.0
        self.theta = math.pi / 2
        self.s = [self.x, self.y, self.theta]
        self.s_prime = []
        self.r = 0  # 看boss是否出界
        # self.s_train = []

    def reset(self, range_x_1=0.2, range_x_2=0.8, range_y=0.8):
        self.x = random.uniform(range_x_1 * MAX_X, range_x_2 * MAX_X)
        self.y = range_y * MAX_Y
        self.theta = -math.pi / 2
        self.s = [self.x, self.y, self.theta]
        # return self.s

    def observe(self, s_1):
        d = math.sqrt(math.pow(self.s[0] - s_1[0], 2) + math.pow(self.s[1] - s_1[1], 2)) + 0.000001
        q_r = math.acos(
            ((s_1[0] - self.s[0]) * math.cos(self.s[2]) + (s_1[1] - self.s[1]) * math.sin(self.s[2])) / d)
        q_b = math.acos(
            ((self.s[0] - s_1[0]) * math.cos(s_1[2]) + (self.s[1] - s_1[1]) * math.sin(s_1[2])) / d)
        return [q_r, q_b, d]

    def observe_prime(self, s_1):
        d = math.sqrt(math.pow(self.s_prime[0] - s_1[0], 2) + math.pow(self.s_prime[1] - s_1[1], 2)) + 0.000001
        q_r = math.acos(
            ((s_1[0] - self.s_prime[0]) * math.cos(self.s_prime[2]) + (s_1[1] - self.s_prime[1]) * math.sin(
                self.s_prime[2])) / d)
        q_b = math.acos(
            ((self.s_prime[0] - s_1[0]) * math.cos(s_1[2]) + (self.s_prime[1] - s_1[1]) * math.sin(s_1[2])) / d)
        return [q_r, q_b, d]

    def select_action(self, s_1_prime):
        t_max = -1
        a = []
        for i in range(3):
            for j in range(3):
                a_b = [i - 1, j - 1]
                # s_b_prime = utils.step(boss_s, a_b)
                self.s_prime = self.step(a_b)
                s = self.observe_prime(s_1_prime)
                t = (MAX_D + MIN_D - 2 * s[2]) / (MAX_D - MIN_D)
                if t > t_max:
                    t_max = t
                    a = a_b
        return a

    def step(self, a):  # 预测做出动作后的之后状态
        v = (a[0] + 1) * MAX_V / 2
        omega = a[1] * MAX_OMEGA
        s_prime = copy.deepcopy(self.s)
        for i in range(int(T / 0.01)):
            s_prime[0] += v * math.cos(s_prime[2]) * 0.1
            s_prime[1] += v * math.sin(s_prime[2]) * 0.1
            s_prime[2] += omega * 0.01
        if s_prime[2] > math.pi:
            s_prime[2] -= 2 * math.pi
        if s_prime[2] < -math.pi:
            s_prime[2] += 2 * math.pi
        return s_prime

    def reward(self):
        if self.s[0] < 0 or self.s[0] > MAX_X or self.s[1] < 0 or self.s[1] > MAX_Y:
            self.r = -1
        # if self.s_train[2] < MIN_D and self.s_train[0] + self.s_train[1] > math.radians(140):
        #     # math.radians()从度转换为弧度
        #     if self.s_train[0] < math.radians(30) < self.s_train[1]:
        #         self.r = 1
        #     if self.s_train[1] < math.radians(30) < self.s_train[0]:
        #         self.r = -1


class train_play():  # 期望为多智体的类调用等，待调整
    def __init__(self, p1=players(), p2=players(), bo=boss()):
        # super(boss_play,self).__init__()
        pygame.init()
        self.p1 = p1
        self.p2 = p2
        self.bo = bo

    def reset(self):
        self.p1.reset()
        self.p2.reset(range_x_1=0.4, range_x_2=0.8)
        # self.p2.reset()
        self.bo.reset()

    def observe(self):
        pass

    def step(self, a):
        pass


class boss_play():  # 期望为多智体的类调用等，待调整
    def __init__(self, p1=players(), p2=players(), bo=boss()):
        # super(boss_play,self).__init__()
        pygame.init()
        self.p1 = p1
        self.p2 = p2
        self.bo = bo
        self.WIDTH, self.HEIGHT = MAX_X, MAX_Y
        self.WINDOW = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.WINDOW = pygame.display.set_mode([self.WIDTH, self.HEIGHT])
        self.bg = load_background_file()
        self.player1_row = load_player1_file()
        self.player2_row = load_player2_file()
        self.boss_row = load_boss_file()
        self.player1 = pygame.transform.rotate(self.player1_row, change_theta(self.p1.theta))
        self.player2 = pygame.transform.rotate(self.player2_row, change_theta(self.p2.theta))
        self.boss = pygame.transform.rotate(self.boss_row, change_theta(self.bo.theta))
        self.player1_rect = self.player1.get_rect()
        self.player2_rect = self.player2.get_rect()
        self.boss_rect = self.boss.get_rect()
        self.render_on = False

    def reset(self):
        self.p1.reset()
        self.p2.reset(range_x_1=0.4, range_x_2=0.8)
        # self.p2.reset()
        self.bo.reset()
        # self.player1 = pygame.transform.rotate(self.player1,change_theta(self.p1.theta))
        # self.player2 = pygame.transform.rotate(self.player2,change_theta(self.p2.theta))
        # self.boss = pygame.transform.rotate(self.boss,change_theta(self.bo.theta))
        # self.player1_rect.centerx, self.player1_rect.centery = self.p1.s[0], self.p1.s[1]
        # self.player2_rect.centerx, self.player2_rect.centery = self.p2.s[0], self.p2.s[1]
        # self.boss_rect.centerx, self.boss_rect.centery = self.bo.s[0], self.bo.s[1]

    def update(self):
        self.player1 = pygame.transform.rotate(self.player1_row, change_theta(self.p1.s[2]))
        self.player2 = pygame.transform.rotate(self.player2_row, change_theta(self.p2.s[2]))
        self.boss = pygame.transform.rotate(self.boss_row, change_theta(self.bo.s[2]))
        # print(change_theta(self.bo.s[2]))
        # time.sleep(0.01)
        self.player1_rect.centerx, self.player1_rect.centery = self.p1.s[0], MAX_Y - self.p1.s[1]
        self.player2_rect.centerx, self.player2_rect.centery = self.p2.s[0], MAX_Y - self.p2.s[1]
        self.boss_rect.centerx, self.boss_rect.centery = self.bo.s[0], MAX_Y - self.bo.s[1]
        # self.player1_rect.centerx, self.player1_rect.centery = self.p1.s[0], self.p1.s[1]
        # self.player2_rect.centerx, self.player2_rect.centery = self.p2.s[0], self.p2.s[1]
        # self.boss_rect.centerx, self.boss_rect.centery = self.bo.s[0], self.bo.s[1]

    def enable_render(self):
        # self.WINDOW = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.render_on = True
        self.draw()

    def draw(self):
        # self.WINDOW.blit(self.bg,self.bg.get_rect())
        self.WINDOW.fill((0, 0, 0))
        self.WINDOW.blit(self.player1, self.player1_rect)
        self.WINDOW.blit(self.player2, self.player2_rect)
        self.WINDOW.blit(self.boss, self.boss_rect)

    def render(self):
        # if not self.render_on:
        #     self.enable_render()
        # else:
        self.update()
        self.draw()
        pygame.display.update()

    def observe(self):
        pass

    def step(self, a):
        pass


def change_theta(theta):
    return theta * 180 / math.pi


def reset():  # 各类初始化总函数
    p1 = players()
    bo = boss()
    return p1.reset(), bo.reset()


def normalize(s):  # 均值化s
    s[0] = (2 * s[0] - MAX_THETA) / MAX_THETA
    s[1] = (2 * s[1] - MAX_THETA) / MAX_THETA
    s[2] = (2 * s[2] - MAX_D) / MAX_D


def render(p1, p2, boss):
    pygame.init()
    pygame.display.set_mode((400, 400))
    pygame.display.set_caption('2v1')


if __name__ == "__main__":
    # p1 = players()
    # p2 = players()
    # bo = boss()
    play = boss_play()
    play.reset()
    FPS = 60
    clock = pygame.time.Clock()
    while True:
        play.bo.a = play.bo.select_action(play.p1.s)
        # print(play.bo.a)
        play.bo.s = play.bo.step(play.bo.a)
        clock.tick(FPS)
        play.p2.a = play.p2.select_action(play.bo.s)
        play.p2.s = play.p2.step(play.p2.a)
        play.p2.s_train = play.p2.observe(play.bo.s)
        play.p1.s_train = play.p1.observe(play.bo.s)
        play.bo.s_train = play.p1.observe(play.p2.s)
        play.p1.reward_test()
        play.p2.reward_test()
        # print(play.bo.s)
        if play.p2.r == 1 or play.bo.r == -1:
            print(play.p2.r)
            print(play.p2.s)
            print(play.p2.s_train)
            print(math.degrees(play.p2.s_train[0]), math.degrees(play.p2.s_train[1]))
            break
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()
        play.render()
