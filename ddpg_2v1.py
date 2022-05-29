import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
from collections import deque
import random
import my_utils
import numpy as np
# import threat
import time
import matplotlib.pyplot as plt
import my_ppo

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 迭代次数(10000)
M = 2000
# 迷你批的大小(64)
N = 100
# 更新网络的次数(50)
nb_train_steps = 50
# 折扣因子(0.99)
gamma = 0.99
# 更新目标网络的系数(0.001)
tau = 0.001


class Critic(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(s_dim, 40)
        self.fc2 = nn.Linear(40 + a_dim, 30)
        self.fc3 = nn.Linear(30, 1)

    def forward(self, s, a):
        s = F.relu(self.fc1(s))
        # torch.cat((x,x),dim=0) dim=0是最终加行， dim=1是最终加列
        q = F.relu(self.fc2(torch.cat([s, a], dim=1)))
        q = self.fc3(q)
        return q


class Actor(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(s_dim, 40)
        self.fc2 = nn.Linear(40, 30)
        self.fc3 = nn.Linear(30, a_dim)

    def forward(self, s):
        a = F.relu(self.fc1(s))
        a = F.relu(self.fc2(a))
        a = torch.tanh(self.fc3(a))
        return a


class Actor_team(nn.Module):
    def __init__(self, s_dim, a_dim, num=2):
        super(Actor_team, self).__init__()
        self.fc1 = nn.Linear(s_dim * num, 40)
        self.fc2 = nn.Linear(40, 30)
        self.fc3 = nn.Linear(30, a_dim)

    def forward(self, s):
        a = F.relu(self.fc1(s))
        a = F.relu(self.fc2(a))
        a = torch.tanh(self.fc3(a))
        return a


class Critic_team(nn.Module):
    def __init__(self, s_dim, a_dim, num=2):
        super(Critic_team, self).__init__()
        self.fc1 = nn.Linear(s_dim * num, 40)
        self.fc2 = nn.Linear(40 + a_dim, 30)
        self.fc3 = nn.Linear(30, 1)

    def forward(self, s, a):
        s = F.relu(self.fc1(s))
        # torch.cat((x,x),dim=0) dim=0是最终加行， dim=1是最终加列
        q = F.relu(self.fc2(torch.cat([s, a], dim=1)))
        q = self.fc3(q)
        return q


class Buffer:
    def __init__(self, size_max):
        self.buffer = deque(maxlen=size_max)

    def store(self, transition):
        self.buffer.append(transition)

    def sample(self):
        minibatch = random.sample(self.buffer, N)
        s_lst = []
        a_lst = []
        r_lst = []
        s_prime_lst = []
        for transition in minibatch:
            s, a, r, s_prime = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append([r])
            s_prime_lst.append(s_prime)
        return torch.Tensor(s_lst).to(device), torch.Tensor(a_lst).to(device), torch.Tensor(r_lst).to(
            device), torch.Tensor(s_prime_lst).to(device)

    def size(self):
        return len(self.buffer)


def select_action(s, actor, method='DDPG'):
    # a_temp = actor(torch.Tensor(s).to(device))
    a = [-1, 0]
    if not actor:
        return a
    if method == 'DDPG':
        a = [actor(torch.Tensor(s).to(device))[0].item(), actor(torch.Tensor(s).to(device))[1].item()]
        a += np.random.normal(loc=0, scale=0.2, size=2)
        a = np.clip(a, -1, 1)
    elif method == 'PPO':
        mu, sigma = actor(torch.tensor([s], dtype=torch.float))
        dis = torch.distributions.normal.Normal(mu, sigma)  # 构建分布
        a = dis.sample()  # 采样出一个动作
        a = torch.tanh(a)
        a = torch.clamp(a, -1, 1)
        a = [a[0][0].item(), a[0][1].item()]
    return a


# def select_action_team(s1, s2, actor):  # 加入队友的状态
#     s = torch.cat((torch.Tensor(s1), torch.Tensor(s2)), dim=0)
#     a_temp = actor(s.to(device))
#     a = [actor(s.to(device))[0].item(), actor(s.to(device))[1].item()]
#     a += np.random.normal(loc=0, scale=0.2, size=2)
#     a = np.clip(a, -1, 1)
#     return a


def soft_update(net, net_target):
    for param, param_target in zip(net.parameters(), net_target.parameters()):
        param_target.data.copy_(tau * param.data + (1 - tau) * param_target.data)


def train_Buffer(buffer, critic, critic_target, critic_optim, actor, actor_target, actor_optim, sum):
    s_lst, a_lst, r_lst, s_prime_lst = buffer.sample()

    target = r_lst + gamma * critic_target(s_prime_lst, actor_target(s_prime_lst))

    critic_loss = F.mse_loss(critic(s_lst, a_lst), target)
    critic_optim.zero_grad()
    critic_loss.backward()
    critic_optim.step()

    actor_loss = -critic(s_lst, actor(s_lst)).mean()
    actor_optim.zero_grad()
    actor_loss.backward()
    actor_optim.step()

    soft_update(critic, critic_target)
    soft_update(actor, actor_target)

    sum[0] += critic_loss
    sum[1] += actor_loss


def draw_pic(loss, num=''):
    plt.figure(1)
    plt.plot(loss[0], label='critic' + num)
    plt.plot(loss[1], label='actor' + num)
    plt.xlabel('episodes')
    plt.ylabel('loss')
    plt.legend()
    plt.show()


def give_net(method, num=1):
    if method == '':
        actor = None
        critic = None
        return actor, critic
    elif method == 'PPO':
        actor = my_ppo.Pi_net()
        critic = my_ppo.V_net()
    elif num == 1:
        critic = Critic(s_dim=3, a_dim=2).to(device)
        actor = Actor(s_dim=3, a_dim=2).to(device)
    else:
        critic = Critic_team(s_dim=3, a_dim=2).to(device)
        actor = Actor_team(s_dim=3, a_dim=2).to(device)
    actor.eval()
    critic.eval()
    return actor, critic


def load_net(method, actor, critic, num=1, train=True, name=1):
    if method == 'PPO':
        actor.load_state_dict(torch.load('pi.pth'))
        if train:
            critic.load_state_dict(torch.load('vi.pth'))
    elif num == 1:
        if train:
            critic.load_state_dict(torch.load('critic.pth'))
            actor.load_state_dict(torch.load('actor.pth'))
        else:
            actor.load_state_dict(torch.load('actor_all_pi/attacter/actor_ddpg.pth'))
    else:
        if train:
            critic.load_state_dict(torch.load('critic_team.pth'))
        actor.load_state_dict(torch.load('actor_team.pth'))


def load_net2(method, actor, critic, num=1, train=True, name=1):
    if method == 'PPO':
        actor.load_state_dict(torch.load('pi.pth'))
        if train:
            critic.load_state_dict(torch.load('vi.pth'))
    elif num == 1:
        if train:
            critic.load_state_dict(torch.load('critic2.pth'))
        actor.load_state_dict(torch.load('actor2.pth'))
    else:
        if train:
            critic.load_state_dict(torch.load('critic2_team.pth'))
        actor.load_state_dict(torch.load('actor2_team.pth'))


def save_net(method, actor, critic, num=1, name=''):
    if method == 'PPO':
        torch.save(actor.state_dict(), 'pi%s.pth' % name)
        torch.save(critic.state_dict(), 'v%s.pth' % name)
    elif num == 1:
        torch.save(actor.state_dict(), 'actor%s.pth' % name)
        torch.save(critic.state_dict(), 'critic%s.pth' % name)
    else:
        torch.save(actor.state_dict(), 'actor%s_team.pth' % name)
        torch.save(critic.state_dict(), 'critic%s_team.pth' % name)


def reward_dis(d, t):
    if d > 100:
        t = t - 0.007 * d
    elif d > 80:
        t = t - 0.005 * d
    elif d > 50:
        t = t - 0.003 * d
    return t


def main1(method1='DDPG', train1=True, method2='DDPG', train2=True, way=0, load1=False, load2=False, num1=1, num2=1,
          dis=False,D_Reward=False, run=False):
    # 网络使用的算法，是否训练, boss的攻击方式，是否加载已存在网络，可看见的状态数量,加入距离回报,分解回报,是否采取逃跑
    actor, critic = give_net(method1, num1)
    actor2, critic2 = give_net(method2, num2)
    if load1:
        load_net(method1, actor, critic, num=num1, train=train1)
    if load2:
        load_net2(method2, actor2, critic2, num=num2, train=train2)
    if train1:
        critic_optim = optim.Adam(critic.parameters())
        actor_optim = optim.Adam(actor.parameters())
        critic_target = copy.deepcopy(critic)
        actor_target = copy.deepcopy(actor)
    if train2:
        critic_optim2 = optim.Adam(critic2.parameters())
        actor_optim2 = optim.Adam(actor2.parameters())
        critic_target2 = copy.deepcopy(critic2)
        actor_target2 = copy.deepcopy(actor2)

    buffer = Buffer(size_max=int(1e6))
    buffer2 = Buffer(size_max=int(1e6))

    loss = [[], []]
    loss2 = [[], []]

    for i in range(M):

        # 第i次迭代的开始时间
        start = time.time()
        p1 = my_utils.players()
        p2 = my_utils.players()
        bo = my_utils.boss()
        p1.reset()
        p2.reset(range_x_1=0.4, range_x_2=0.8)
        bo.reset()
        p1.s_train = p1.observe(bo.s)
        p2.s_train = p2.observe(bo.s)
        p1.normalize(p1.s_train)
        p2.normalize(p2.s_train)
        for j in range(my_utils.N):
            if num1 == 1:
                # a_r = select_action(p1.s_train, actor, method=method1)
                a_r = [-1, 0]
            else:
                if method1 == 'PPO':
                    a_r = select_action(p1.s_train, actor, method=method1)
                else:
                    a_r = select_action(p1.s_train + p2.s_train, actor)
                # a_npc = select_action(p2.s_train, actor2, method=method2)
            if num2 == 1:
                a_npc = select_action(p2.s_train, actor2, method=method2)
            else:
                # if method1 == 'PPO':
                #     a_r = select_action(p1.s_train, actor, method=method1)
                # else:
                #     a_r = select_action(p1.s_train + p2.s_train, actor)
                a_npc = select_action(p2.s_train + p1.s_train, actor2)
            # a_npc = p2.select_action(bo.s)
            if way == 0:
                a_b = bo.select_action(p1.s)
            else:
                a_b = bo.select_action(p2.s)
            s_r_prime = p1.step(a_r)
            s_npc_prime = p2.step(a_npc)
            s_b_prime = bo.step(a_b)
            p1.s_train_prime = p1.observe(s_b_prime)
            p2.s_train_prime = p2.observe(s_b_prime)
            if s_b_prime[0] < 0 or s_b_prime[0] > my_utils.MAX_X or s_b_prime[1] < 0 or s_b_prime[1] > my_utils.MAX_Y:
                p1.r = 10
                p2.r = 10
                if num1 == 1:
                    buffer.store((p1.s_train, a_r, p1.r + p2.r, p1.s_train_prime))
                else:
                    buffer.store((p1.s_train + p2.s_train, a_r, p1.r + p2.r, p1.s_train_prime + p2.s_train_prime))
                if num2 == 1:
                    buffer2.store((p2.s_train, a_npc, p2.r + p1.r, p2.s_train_prime))
                else:
                    buffer2.store((p2.s_train + p1.s_train, a_npc, p2.r + p1.r, p2.s_train_prime + p1.s_train_prime))
                break
            # t1 = p1.evalute4(p1.s_train_prime)
            # t1 = p1.evalute3(p1.s_train_prime)
            t1 = p1.evalute2(j)
            t2 = p2.evalute(p2.s_train_prime)
            # t2 = 0
            if dis:
                d1 = p1.s_train_prime[2]
                d2 = p2.s_train_prime[2]
                t1 = reward_dis(d1, t1)
                t2 = reward_dis(d2, t2)

            p2.reward()
            if run:
                p1.reward2(p2.r)
            else:
                p1.reward()
            bo.reward()
            # utils.normalize(s_prime)
            if train1:
                p1.normalize(p1.s_train_prime)
            if train2:
                p2.normalize(p2.s_train_prime)
            if p1.r == 0 and p2.r == 0 and bo.r == 0:
                if num1 == 1:
                    if j == N - 1 and run:
                        t1 = 10
                    if train1:
                        # print(i)
                        print('t1', t1)
                        buffer.store((p1.s_train, a_r, t1, p1.s_train_prime))
                else:
                    if train1:
                        buffer.store((p1.s_train + p2.s_train, a_r, t1, p1.s_train_prime + p2.s_train_prime))
                if num2 == 1:
                    if train2:
                        # print(t2)
                        print('t2', t2)
                        buffer2.store((p2.s_train, a_npc, t2, p2.s_train_prime))
                else:
                    if train2:
                        buffer2.store((p2.s_train + p1.s_train, a_npc, t2, p2.s_train_prime + p1.s_train_prime))
                p1.s = s_r_prime
                p2.s = s_npc_prime
                bo.s = s_b_prime
                p1.s_train = p1.s_train_prime
                p2.s_train = p2.s_train_prime
            else:
                if D_Reward:
                    if p1.r > 0:
                        t1 = 0.6 * t1 + 0.4 * p1.r
                        t2 = 0.1 * p1.r + 0.9 * t2
                    elif p2.r > 0:  # 鼓励队友2赢得比赛
                        t1 = 0.8 * p2.r + 0.2 * t1
                        t2 = p2.r
                    else:
                        t1 = t2 = -5
                else:
                    t1 = t2 = p1.r + p2.r
                    # t2 = p2.r
                if num1 == 1:
                    if train1:
                        buffer.store((p1.s_train, a_r, t1, p1.s_train_prime))
                else:
                    if train1:
                        buffer.store((p1.s_train + p2.s_train, a_r, t1, p1.s_train_prime + p2.s_train_prime))
                if num2 == 1:
                    if train2:
                        # print(t2)
                        buffer2.store((p2.s_train, a_npc, t2, p2.s_train_prime))
                else:
                    if train2:
                        buffer2.store((p2.s_train + p1.s_train, a_npc, t2, p2.s_train_prime + p1.s_train_prime))
                break

        if buffer.size() >= N or buffer2.size() >= N:

            sum = [0, 0]
            sum2 = [0, 0]

            for j in range(nb_train_steps):
                if train1:
                    train_Buffer(buffer, critic, critic_target, critic_optim, actor, actor_target, actor_optim, sum)
                if train2:
                    train_Buffer(buffer=buffer2, critic=critic2, critic_target=critic_target2,
                                 critic_optim=critic_optim2,
                                 actor=actor2, actor_target=actor_target2, actor_optim=actor_optim2, sum=sum2)

            if train1:
                loss[0].append(sum[0].detach().cpu().numpy() / nb_train_steps)
                loss[1].append(sum[1].detach().cpu().numpy() / nb_train_steps)
            if train2:
                loss2[0].append(sum2[0].detach().cpu().numpy() / nb_train_steps)
                loss2[1].append(sum2[1].detach().cpu().numpy() / nb_train_steps)

        # 第i次迭代的结束时间
        end = time.time()

        print('第', i + 1, '次迭代用时：', end - start, 's', '回报', p1.r, p2.r)

    if train1:
        draw_pic(loss)
        save_net(method=method1, actor=actor, critic=critic, num=num1)
    if train2:
        draw_pic(loss2, '2')
        save_net(method=method2, actor=actor2, critic=critic2, num=num2, name='2')


if __name__ == '__main__':
    main1(method1='DDPG', train1=False, train2=True, method2='DDPG', way=0, num1=1, num2=2, dis=False, D_Reward=False,
          load1=True, load2=False, run=False)
    # critic = Critic_team(s_dim=3, a_dim=2).to(device)
    # critic = Critic(s_dim=3, a_dim=2).to(device)
    # # # critic.load_state_dict(torch.load('critic.pth'))
    # critic.eval()
    # # actor = Actor_team(s_dim=3, a_dim=2).to(device)
    # actor = Actor(s_dim=3, a_dim=2).to(device)
    # # actor = my_ppo.Pi_net()
    # actor.eval()
    # # actor.load_state_dict(torch.load('actor.pth'))
    # # critic2 = Critic_team(s_dim=3, a_dim=2).to(device)
    # critic2 = Critic(s_dim=3, a_dim=2).to(device)
    # # critic2.load_state_dict(torch.load('critic2_2.pth'))
    # critic2.eval()
    #
    # # actor2 = Actor_team(s_dim=3, a_dim=2).to(device)
    # actor2 = Actor(s_dim=3, a_dim=2).to(device)
    # # actor2.load_state_dict(torch.load('actor2.pth'))
    # actor2.eval()
    #
    # # critic2 = copy.deepcopy(critic)
    # # actor2 = copy.deepcopy(actor)
    #
    # critic_optim = optim.Adam(critic.parameters())
    # actor_optim = optim.Adam(actor.parameters())
    # critic_optim2 = optim.Adam(critic2.parameters())
    # actor_optim2 = optim.Adam(actor2.parameters())
    #
    # critic_target = copy.deepcopy(critic)
    # actor_target = copy.deepcopy(actor)
    # critic_target2 = copy.deepcopy(critic2)
    # actor_target2 = copy.deepcopy(actor2)
    #
    # buffer = Buffer(size_max=int(1e6))
    # buffer2 = Buffer(size_max=int(1e6))
    #
    # loss = [[], []]
    # loss2 = [[], []]
    # # actor2 = copy.deepcopy(actor)
    # actor.load_state_dict(torch.load('actor_all_pi/actor_nomove/actor_revolve.pth'))
    # # actor.load_state_dict(torch.load('actor_all_pi/actor_nomove/actor_revolve.pth'))
    # # actor.eval()
    # # actor = my_ppo.Agent()
    # # actor.load()
    # for i in range(M):
    #
    #     # 第i次迭代的开始时间
    #     start = time.time()
    #     p1 = my_utils.players()
    #     p2 = my_utils.players()
    #     bo = my_utils.boss()
    #     p1.reset()
    #     p2.reset(range_x_1=0.4, range_x_2=0.8)
    #     bo.reset()
    #     p1.s_train = p1.observe(bo.s)
    #     p2.s_train = p2.observe(bo.s)
    #     p1.normalize(p1.s_train)
    #     p2.normalize(p2.s_train)
    #     for j in range(my_utils.N):
    #         # print('进来')
    #         # a_r = actor.choose_action(torch.tensor([p1.s_train], dtype=torch.float))
    #         # a_r = select_action(p1.s_train+p2.s_train, actor)
    #         a_r = select_action(p1.s_train, actor)
    #         # a_r = select_action_team(p1.s_train, p2.s_train, actor)
    #         # a_r = [actor(torch.Tensor(p1.s_train).to(device))[0].item(),
    #         #        actor(torch.Tensor(p1.s_train).to(device))[1].item()]
    #         # a_r = [-1, 0]
    #         a_npc = select_action(p2.s_train, actor2)
    #         # a_npc = select_action(p2.s_train + p1.s_train, actor2)
    #         # a_npc = [-1,0]
    #         # a_npc = [actor2(torch.Tensor(p2.s_train).to(device))[0].item(),
    #         #          actor2(torch.Tensor(p2.s_train).to(device))[1].item()]
    #         # a_npc = p2.select_action(bo.s)
    #         a_b = bo.select_action(p1.s)
    #         # s_r_prime = p1.step(a_r)
    #         p1.s = p1.step(a_r)
    #         s_npc_prime = p2.step(a_npc)
    #         s_b_prime = bo.step(a_b)
    #         if s_b_prime[0] < 0 or s_b_prime[0] > my_utils.MAX_X or s_b_prime[1] < 0 or s_b_prime[1] > my_utils.MAX_Y:
    #             print('会有这种情况吗')
    #             break
    #         # s_prime = utils.observe(s_r_prime, s_b_prime)
    #         p1.s_train = p1.observe(s_b_prime)
    #         # p1.s_train = p1.observe(bo.s)
    #         # p1.s_train_prime = p1.observe(s_b_prime)
    #         # p2.s_train = p2.observe(s_b_prime)
    #         p2.s_train_prime = p2.observe(s_b_prime)
    #         # p2.s_train = p2.observe(s_b_prime)
    #         # t = utils.evaluate(s_prime)
    #         # d1 = p1.s_train_prime[2]
    #         # d2 = p2.s_train_prime[2]
    #         # t1 = p1.evalute(p1.s_train_prime)
    #         t2 = p2.evalute(p2.s_train_prime)
    #         # r = utils.reward(s_prime)
    #         # p1.reward()
    #         p1.reward_test()
    #         # p2.reward_test()
    #         p2.reward()
    #         # bo.reward()
    #         # utils.normalize(s_prime)
    #         # p1.normalize(p1.s_train_prime)
    #         p2.normalize(p2.s_train_prime)
    #         # p2.r = p1.r + p2.r
    #         # if d1 > 150:
    #         #     t1 = t1 - 0.007 * d1
    #         # elif d1 > 100:
    #         #     t1 = t1 - 0.005 * d1
    #         # elif d1 > 30:
    #         #     t1 = t1 - 0.003 * d1
    #         # else:
    #         #     pass
    #         # if d2 > 150:
    #         #     t2 = t2 - 0.007 * d2
    #         # elif d2 > 100:
    #         #     t2 = t2 - 0.005 * d2
    #         # elif d2 > 30:
    #         #     t2 = t2 - 0.003 * d2
    #         # else:
    #         #     pass
    #         if p2.r == 0 and p1.r == 0:
    #             # buffer.store((p1.s_train, a_r, t1, p1.s_train_prime))
    #             # buffer.store((p1.s_train + p2.s_train, a_r, t1, p1.s_train_prime+p2.s_train_prime))
    #             buffer2.store((p2.s_train, a_npc, t2, p2.s_train_prime))
    #             # buffer.store((p2.s_train, a_npc, t2, p2.s_train_prime))
    #             # buffer2.store((p2.s_train + p1.s_train, a_npc, t2, p2.s_train_prime+p1.s_train_prime))
    #             # p1.s = s_r_prime
    #             p2.s = s_npc_prime
    #             bo.s = s_b_prime
    #             # p1.s_train = p1.s_train_prime
    #             p2.s_train = p2.s_train_prime
    #         else:
    #             # buffer.store((p1.s_train, a_r, p1.r + p2.r, p1.s_train_prime))
    #             # buffer.store((p1.s_train+p2.s_train, a_r, p1.r + p2.r, p1.s_train_prime+p2.s_train_prime))
    #             buffer2.store((p2.s_train, a_npc, p2.r + p1.r, p2.s_train_prime))
    #             # buffer.store((p2.s_train, a_npc, p2.r + p1.r, p2.s_train_prime))
    #             # buffer2.store((p2.s_train+p1.s_train, a_npc, p2.r + p1.r, p2.s_train_prime+p1.s_train_prime))
    #             break
    #
    #     if buffer2.size() >= N or buffer.size() >= N:
    #
    #         # sum = [0, 0]
    #         sum2 = [0, 0]
    #
    #         for j in range(nb_train_steps):
    #             # train_Buffer(buffer, critic2, critic_target2, critic_optim2, actor2, actor_target2, actor_optim2, sum)
    #             # train_Buffer(buffer, critic, critic_target, critic_optim, actor, actor_target, actor_optim, sum)
    #             train_Buffer(buffer2, critic2, critic_target2, critic_optim2, actor2, actor_target2, actor_optim2, sum2)
    #             # s_lst, a_lst, r_lst, s_prime_lst = buffer2.sample()
    #             #
    #             # target = r_lst + gamma * critic_target2(s_prime_lst, actor_target2(s_prime_lst))
    #             #
    #             # critic_loss = F.mse_loss(critic2(s_lst, a_lst), target)
    #             # critic_optim2.zero_grad()
    #             # critic_loss.backward()
    #             # critic_optim2.step()
    #             #
    #             # actor_loss = -critic2(s_lst, actor2(s_lst)).mean()
    #             # actor_optim2.zero_grad()
    #             # actor_loss.backward()
    #             # actor_optim2.step()
    #             #
    #             # soft_update(critic2, critic_target2)
    #             # soft_update(actor2, actor_target2)
    #             #
    #             # sum2[0] += critic_loss
    #             # sum2[1] += actor_loss
    #         # loss[0].append(sum[0].detach().cpu().numpy() / nb_train_steps)
    #         # loss[1].append(sum[1].detach().cpu().numpy() / nb_train_steps)
    #         loss2[0].append(sum2[0].detach().cpu().numpy() / nb_train_steps)
    #         loss2[1].append(sum2[1].detach().cpu().numpy() / nb_train_steps)
    #
    #     # 第i次迭代的结束时间
    #     end = time.time()
    #
    #     print('第', i + 1, '次迭代用时：', end - start, 's','回报',p1.r+p2.r)
    #
    # # draw_pic(loss)
    # draw_pic(loss2, '2')
    #
    # # torch.save(critic.state_dict(), 'critic_team.pth')
    # # torch.save(actor.state_dict(), 'actor_team.pth')
    # torch.save(critic2.state_dict(), 'critic2.pth')
    # torch.save(actor2.state_dict(), 'actor2.pth')
