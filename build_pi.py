from manule_control import choose_action, choose_actor
import test_ddpg_2v1 as tg
import torch
import my_ddpg
import ddpg_2v1
import my_ppo
import numpy as np


def build_strategy():
    ac1 = choose_action(name="actor_pi")  # 玩家策略
    ac2 = choose_actor(name="actor_pi")  # 队友策略
    strategy = np.zeros([len(ac1), len(ac2)])
    actor_on = ddpg_2v1.Actor(s_dim=3, a_dim=2).to(my_ddpg.device)  # 独自对抗
    actor_ti = ddpg_2v1.Actor_team(s_dim=3, a_dim=2).to(my_ddpg.device)  # 可以观测队友的对抗
    actor_ppo = my_ppo.Pi_net()  # PPO对抗
    actor2_on = ddpg_2v1.Actor(s_dim=3, a_dim=2).to(my_ddpg.device)  # 独自对抗
    actor2_ti = ddpg_2v1.Actor_team(s_dim=3, a_dim=2).to(my_ddpg.device)  # 可以观测队友的对抗
    actor_on.eval()
    actor_ti.eval()
    actor_ppo.eval()
    actor2_on.eval()
    actor2_ti.eval()
    row = 0
    # col = 0
    for a1 in ac1:
        print("当前a1的策略为", a1)
        col = 0
        try:
            actor1 = actor_on
            actor1.load_state_dict(torch.load(a1))
            method1 = "DDPG"
            num1 = 1
        except RuntimeError:
            try:
                actor1 = actor_ti
                actor1.load_state_dict(torch.load(a1))
                method1 = "DDPG"
                num1 = 2
            except RuntimeError:
                actor1 = actor_ppo
                actor1.load_state_dict(torch.load(a1))
                method1 = "PPO"
                num1 = 1
        for a2 in ac2:
            sum_r = 0
            print("当前a2的策略为", a2)
            try:
                actor2 = actor2_on
                actor2.load_state_dict(torch.load(a2))
                method2 = "DDPG"
                num2 = 1
            except RuntimeError:
                actor2 = actor2_ti
                actor2.load_state_dict(torch.load(a2))
                method2 = "DDPG"
                num2 = 2
            for i in range(100):
                list_0, list_1, list_2, r = tg.test_no_render(
                    actor=actor1, method1=method1, num1=num1, actor2=actor2, method2=method2, num2=num2, way=0)
                if r > 0:
                    sum_r += 1
            strategy[row][col] = sum_r / 100
            col += 1
        row += 1
    print(strategy)
    save_np(arr=strategy)


def save_np(file="strategy", arr=np.zeros([7, 7])):
    np.save(file=file, arr=arr)


def load_np(file="strategy.npy"):
    ff = np.load(file=file)
    print(ff)
    return ff


def avg_np(file="strategy.npy"):
    ff = load_np(file=file)
    avgs = []
    for i in ff:
        sum = 0
        for j in i:
            sum += j
        avg = sum / len(i)
        avgs.append(round(avg, 2))
    print(avgs)
    return avgs


def style_strategy(strategy, num=3):  # 计算风格的平均胜率,每个风格三个策略
    time = 0
    all_styles = int(len(strategy) / 3)
    avg_style = np.zeros([all_styles,len(strategy[0])])
    for i in range(all_styles):
        for j in range(len(strategy[0])):
            sum = 0
            now = time
            for k in range(num):
                sum += strategy[now][j]
                now += 1
            avg_style[i][j] = round(sum/num,2)
        time += num
    print(avg_style)
    save_np(file="style_strategy",arr=avg_style)
    return avg_style


if __name__ == "__main__":
    # build_strategy()
    ac1 = choose_action(name="actor_pi")  # 玩家策略
    ac2 = choose_actor(name="actor_pi")  # 队友策略
    strategy = load_np()
    style_strategy(strategy)
