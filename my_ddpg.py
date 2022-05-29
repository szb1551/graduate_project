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

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 迭代次数(10000)
M = 2000
# 迷你批的大小(64)
N = 64
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


def select_action(s, actor):
    a = [actor(torch.Tensor(s).to(device))[0].item(), actor(torch.Tensor(s).to(device))[1].item()]
    # print(a)
    a += np.random.normal(loc=0, scale=0.2, size=2)
    a = np.clip(a, -1, 1)
    return a


def soft_update(net, net_target):
    for param, param_target in zip(net.parameters(), net_target.parameters()):
        param_target.data.copy_(tau * param.data + (1 - tau) * param_target.data)


if __name__ == '__main__':

    critic = Critic(s_dim=3, a_dim=2).to(device)
    actor = Actor(s_dim=3, a_dim=2).to(device)

    critic_optim = optim.Adam(critic.parameters())
    actor_optim = optim.Adam(actor.parameters())

    critic_target = copy.deepcopy(critic)
    actor_target = copy.deepcopy(actor)

    buffer = Buffer(size_max=int(1e6))

    loss = [[], []]

    for i in range(M):

        # 第i次迭代的开始时间
        start = time.time()

        # s_r, s_b = utils.reset()
        p1 = my_utils.players()
        p2 = my_utils.players()
        bo = my_utils.boss()
        # p1.reset()
        p1.reset(range_x_1=0.4, range_x_2=0.8)
        bo.reset()
        p2.reset()
        # s = utils.observe(s_r, s_b)
        p1.s_train = p1.observe(bo.s)
        # print(p1.s_train)
        # utils.normalize(s)
        p1.normalize(p1.s_train)
        # s = p1.s_train
        # print(s)
        for j in range(my_utils.N):
            a_r = select_action(p1.s_train, actor)
            a_npc = [-1, 0]
            # a_b = threat.select_action(s_r, s_b)
            a_b = bo.select_action(p2.s)
            # print(s_r)
            # print(_r)
            # s_r_prime = utils.step(s_r, a_r)
            s_r_prime = p1.step(a_r)
            # print(s_r_prime)
            if s_r_prime[0] < 0 or s_r_prime[0] > my_utils.MAX_X or s_r_prime[1] < 0 or s_r_prime[1] > my_utils.MAX_Y:
                break
            # s_b_prime = utils.step(s_b, a_b)
            s_b_prime = bo.step(a_b)
            # print(s_b_prime)
            if s_b_prime[0] < 0 or s_b_prime[0] > my_utils.MAX_X or s_b_prime[1] < 0 or s_b_prime[1] > my_utils.MAX_Y:
                break
            # s_prime = utils.observe(s_r_prime, s_b_prime)
            p2.s_train = p2.observe(s_b_prime)
            p1.s_train_prime = p1.observe(s_b_prime)
            # t = utils.evaluate(s_prime)
            t = p1.evalute(p1.s_train_prime)
            # print(t)
            # r = utils.reward(s_prime)
            p1.reward()
            p2.reward_test()
            # utils.normalize(s_prime)
            p1.normalize(p1.s_train_prime)
            if p1.r == 0 and p2.r == 0:
                buffer.store((p1.s_train, a_r, t, p1.s_train_prime))
                # s_r = s_r_prime
                p1.s = s_r_prime
                # s_b = s_b_prime
                bo.s = s_b_prime
                p1.s_train = p1.s_train_prime
            else:
                buffer.store((p1.s_train, a_r, p1.r+p2.r, p1.s_train_prime))
                break

        if buffer.size() >= N:

            sum = [0, 0]

            for j in range(nb_train_steps):
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

            loss[0].append(sum[0].detach().cpu().numpy() / nb_train_steps)
            loss[1].append(sum[1].detach().cpu().numpy() / nb_train_steps)

        # 第i次迭代的结束时间
        end = time.time()

        print('第', i + 1, '次迭代用时：', end - start, 's')

    plt.figure(1)
    plt.plot(loss[0], label='critic')
    plt.plot(loss[1], label='actor')
    plt.xlabel('episodes')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

    torch.save(critic.state_dict(), 'critic2.pth')
    torch.save(actor.state_dict(), 'actor2.pth')
