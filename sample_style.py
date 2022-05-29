import torch
import ddpg_2v1
import my_ppo
import os
import re
import my_ddpg
import my_utils
import matplotlib.pyplot as plt

color_library = ['r', 'g', 'b', 'purple', 'brown', 'black', 'gold']


def all_style(name="actor_pi"):  # 所有的对抗风格
    styles = os.listdir(name)
    return styles


def style_pi(style="对抗", name="actor_pi"):  # 风格的所有策略选取
    ore = re.compile(r'^pi|^actor\D')
    loc = name + '/' + style
    actors = os.listdir(loc)  # 风格里的所有策略
    style_actors = []
    for ac in actors:
        if ore.search(ac):
            temp = loc + '/' + ac
            style_actors.append(temp)
    return style_actors


def sample_action(actor=None, method1='DDPG', actor2=None, method2='DDPG', way=0, num1=1, num2=1):  # 采集动作
    env = my_utils.train_play()
    env.reset()
    done = False
    play_T = -1  # 采集时间
    action_list = []
    env.p1.s_train = env.p1.observe(env.bo.s)
    env.p2.s_train = env.p2.observe(env.bo.s)
    times = 0  # 1分钟内解决
    while not done and times < 60:
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
        if actor2:
            if num2 == 1:
                if method2 == 'DDPG':
                    env.p2.a = [actor2(torch.Tensor(env.p2.s_train).to(my_ddpg.device))[0].item(),
                                actor2(torch.Tensor(env.p2.s_train).to(
                                    my_ddpg.device))[1].item()]
                elif method2 == 'PPO':
                    env.p2.a = ddpg_2v1.select_action(env.p2.s_train, actor=actor2, method=method2)
            else:
                env.p2.a = ddpg_2v1.select_action(env.p2.s_train + env.p1.s_train, actor=actor2, method=method2)
        else:
            env.p2.a = [-1, 0]
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
        play_T += 1
        if play_T > 15:
            times += 1
            action_list.append(env.p1.a)
            play_T = 0

        if env.p1.r + env.p2.r != 0:
            action_list.append(env.p1.a)
            print(action_list)
            done = True
    return action_list


def draw_action(action_list, c=None):  # 绘制v，w采样图
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


def draw_actions(action_lists):  # 绘制不同风格的v,w采样图
    labels = ['combat','delay','die','run']
    color_choose = 0
    for action_list in action_lists:
        print(action_list)
        v_list = []
        w_list = []
        for ac in action_list:
            v_list.append(ac[0])
            w_list.append(ac[1])
        plt.scatter(w_list, v_list, c=color_library[color_choose])
        color_choose += 1
    plt.xlim(-1, 1)
    plt.ylim(-1.5, 1.5)
    plt.xlabel('w')
    plt.ylabel('v')
    plt.legend(labels=labels[:color_choose], loc='best')
    plt.show()


def style_action(style="对抗", name="actor_pi", show=False):  # 风格的动作选取
    style_actors = style_pi(style=style, name=name)
    actor_on = ddpg_2v1.Actor(s_dim=3, a_dim=2).to(my_ddpg.device)  # 独自对抗
    actor_ti = ddpg_2v1.Actor_team(s_dim=3, a_dim=2).to(my_ddpg.device)  # 可以观测队友的对抗
    actor_ppo = my_ppo.Pi_net()  # PPO对抗
    actor_on.eval()
    actor_ti.eval()
    actor_ppo.eval()
    action_list = []
    for ac in style_actors:
        try:
            actor1 = actor_on
            method1 = "DDPG"
            num1 = 1
            actor1.load_state_dict(torch.load(ac))
        except RuntimeError:
            try:
                actor1 = actor_ti
                method1 = "DDPG"
                num1 = 2
                actor1.load_state_dict(torch.load(ac))
            except RuntimeError:
                actor1 = actor_ppo
                method1 = "PPO"
                num1 = 1
                actor1.load_state_dict(torch.load(ac))
        a_list = sample_action(actor=actor1, num1=num1, method1=method1)
        action_list.extend(a_list)
        if show:
            draw_action(a_list, c='purple')
    return action_list


def main():
    styles = all_style()
    action_lists = []
    for style in styles:
        action_list = style_action(style=style)
        action_lists.append(action_list)
    draw_actions(action_lists)


if __name__ == "__main__":
    main()
