"""
@ Author: Peter Xiao
@ Date: 2020/7/23
@ Filename: Actor_critic.py
@ Brief: 使用 Actor-Critic算法训练CartPole-v0
"""

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import my_utils

# Hyper Parameters for Actor
# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 迭代次数(10000)
M = 100
# 迷你批的大小(64)
N = 64
# 更新网络的次数(50)
nb_train_steps = 50
# 更新目标网络的系数(0.001)
tau = 0.001
# 折扣因子
GAMMA = 0.95  # discount factor
# 学习速率
LR = 0.001  # learning rate

# Use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False  # 非确定性算法


class PGNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PGNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = torch.tanh(self.fc3(out))
        return out

    def initialize_weights(self):  # normalize
        for m in self.modules():
            nn.init.normal_(m.weight.data, 0, 0.1)
            nn.init.constant_(m.bias.data, 0.01)


class Actor(object):
    # dqn Agent
    def __init__(self, env):  # 初始化
        # 状态空间和动作空间的维度
        # self.state_dim = env.observation_space.shape[0]
        self.state_dim = len(env.s_space)
        # self.action_dim = env.action_space.n
        self.action_dim = len(env.a_space)
        # init network parameters
        self.network = PGNetwork(state_dim=self.state_dim, action_dim=self.action_dim).to(device)  # policy网络
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LR)

        # init some parameters
        self.time_step = 0

    def choose_action(self, observation):  # 将观察值放入，返回动作
        observation = torch.FloatTensor(observation).to(device)
        network_output = self.network.forward(observation)  # 网络的输出
        # with torch.no_grad():
        #     prob_weights = F.softmax(network_output, dim=0).cuda().data.cpu().numpy()
        # # prob_weights = F.softmax(network_output, dim=0).detach().numpy()
        # action = np.random.choice(range(prob_weights.shape[0]),
        #                           p=prob_weights)  # select action w.r.t the actions prob
        # a = [actor(torch.Tensor(s).to(device))[0].item(), actor(torch.Tensor(s).to(device))[1].item()]
        # print(network_output)
        a = [network_output[0].item(), network_output[1].item()]
        # print(a)
        a += np.random.normal(loc=0, scale=0.2, size=2)
        a = np.clip(a, -1, 1)
        return a

    def select_action(self, s):
        a = [self.network(torch.Tensor(s).to(device))[0].item(), self.network(torch.Tensor(s).to(device))[1].item()]
        print(a)
        a += np.random.normal(loc=0, scale=0.2, size=2)
        a = np.clip(a, -1, 1)
        return a

    def learn(self, state, action, td_error):  # 学习与更新网络
        self.time_step += 1
        # Step 1: 前向传播
        # softmax_input = self.network.forward(torch.FloatTensor(state).to(device)).unsqueeze(0)
        softmax_input = self.network.forward(torch.FloatTensor(state).to(device))
        # action = torch.LongTensor([action]).to(device)
        # cross_entropy 先softmax再log最后取nll_loss
        # soft_out = F.softmax(out)
        # log_soft_out = torch.log(soft_out)
        # loss = F.nll_loss(log_soft_out, y)
        # print(action)
        print('----------------------')
        print(softmax_input)
        print('---------------------')
        s_input = [softmax_input[0].item(), softmax_input[1].item()]
        print(action)
        print(s_input)
        # neg_log_prob = F.cross_entropy(input=softmax_input, target=action, reduction='none')
        neg_log_prob = (s_input[0] + s_input[1] - action[0] - action[1]).mean()
        # actor_loss = -critic(s_lst, actor(s_lst)).mean()
        # Step 2: 反向传播
        # 这里需要最大化当前策略的价值，因此需要最大化neg_log_prob * tf_error,即最小化-neg_log_prob * td_error
        loss_a = -neg_log_prob * td_error
        self.optimizer.zero_grad()
        loss_a.backward()
        self.optimizer.step()


# Hyper Parameters for Critic
EPSILON = 0.01  # final value of epsilon
REPLAY_SIZE = 10000  # experience replay buffer size
BATCH_SIZE = 64  # size of minibatch
REPLACE_TARGET_FREQ = 100  # frequency to update target Q network


class QNetwork(nn.Module):
    def __init__(self, state_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 1)  # 这个地方和之前略有区别，输出不是动作维度，而是一维

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

    def initialize_weights(self):
        for m in self.modules():
            nn.init.normal_(m.weight.data, 0, 0.1)
            nn.init.constant_(m.bias.data, 0.01)


class Critic(object):
    def __init__(self, env):
        # 状态空间和动作空间的维度
        # self.state_dim = env.observation_space.shape[0]
        self.state_dim = len(env.s_space)
        # self.action_dim = env.action_space.n

        # init network parameters
        self.network = QNetwork(state_dim=self.state_dim).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

        # init some parameters
        self.time_step = 0
        self.epsilon = EPSILON  # epsilon值是随机不断变小的

    def train_Q_network(self, state, reward, next_state):
        s, s_ = torch.FloatTensor(state).to(device), torch.FloatTensor(next_state).to(device)
        # 前向传播
        v = self.network.forward(s)  # v(s)
        v_ = self.network.forward(s_)  # v(s')

        # 反向传播
        loss_q = self.loss_func(reward + GAMMA * v_, v)
        self.optimizer.zero_grad()
        loss_q.backward()
        self.optimizer.step()

        with torch.no_grad():
            td_error = reward + GAMMA * v_ - v

        return td_error


# Hyper Parameters
ENV_NAME = 'CartPole-v0'
EPISODE = 100  # Episode limitation
STEP = 3000  # Step limitation in an episode
TEST = 10  # The number of experiment test every 100 episode


# def test():
#     if episode % 100 == 0:
#         total_reward = 0
#         for i in range(TEST):
#             state = env.reset()
#             for j in range(STEP):
#                 env.render()
#                 action = actor.choose_action(state)  # direct action for test
#                 state, reward, done, _ = env.step(action)
#                 total_reward += reward
#                 if done:
#                     break
#         ave_reward = total_reward / TEST
#         print('episode: ', episode, 'Evaluation Average Reward:', ave_reward)


def main():
    # initialize OpenAI Gym env and dqn agent
    # env = gym.make(ENV_NAME)
    env = my_utils.players()
    env2 = my_utils.players()
    bo = my_utils.boss()
    actor = Actor(env)
    critic = Critic(env)

    for episode in range(EPISODE):
        # initialize task
        # state = env.reset()
        start = time.time()
        env.reset()
        env2.reset()
        bo.reset()
        env.s_train = env.observe(bo.s)
        # Train

        for step in range(STEP):
            # action = actor.choose_action(env.s_train)  # SoftMax概率选择action
            action = actor.choose_action(env.s_train)  # SoftMax概率选择action
            print(action)
            # next_state, reward, done, _ = env.step(action)
            bo.a = bo.select_action(env.s)
            env.s_prime = env.step(action)
            bo.s_prime = bo.step(bo.a)
            env.s_train_prime = env.observe(bo.s)
            # t = env.evalute(env.s_train_prime)
            env.reward()
            td_error = critic.train_Q_network(env.s_train, env.r,
                                              env.s_train_prime)  # gradient = grad[r + gamma * V(s_) - V(s)]
            # actor.learn(state, action, td_error)  # true_gradient = grad[logPi(s,a) * td_error]
            actor.learn(env.s_train, action, td_error)  # true_gradient = grad[logPi(s,a) * td_error]
            env.s_train = env.s_train_prime
            env.s = env.s_prime
            bo.s = bo.s_prime
            if env.r:
                break
        end = time.time()
        print('第', episode + 1, '次迭代用时：', end - start, 's')
    torch.save(critic.network.state_dict(), 'critic.pth')
    torch.save(actor.network.state_dict(), 'actor.pth')


if __name__ == '__main__':
    time_start = time.time()
    main()
    time_end = time.time()
    print('Total time is ', time_end - time_start, 's')
