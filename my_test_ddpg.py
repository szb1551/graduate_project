import my_ddpg
import torch
import my_utils
# import threat
import matplotlib.pyplot as plt


def dd_test(actor):
    # s_r, s_b = utils.reset()
    p1 = my_utils.players()
    bo = my_utils.boss()
    p1.reset()
    bo.reset()
    # s = utils.observe(s_r, s_b)
    p1.s_train = p1.observe(bo.s)
    # list_0 = [[s_r[0], s_r[1]]]
    # list_1 = [[s_b[0], s_b[1]]]
    list_0 = [[p1.s[0],p1.s[1]]]
    list_1 = [[bo.s[0],bo.s[1]]]
    # r = 0
    # print(p1.r)
    for i in range(my_utils.N):
        # utils.normalize(s)
        p1.normalize(p1.s_train)
        a_r = [actor(torch.Tensor(p1.s_train).to(my_ddpg.device))[0].item(), actor(torch.Tensor(p1.s_train).to(
            my_ddpg.device))[1].item()]
        # a_b = threat.select_action(s_r, s_b)
        a_b = bo.select_action(p1.s)
        # s_r = utils.step(s_r, a_r)
        # s_b = utils.step(s_b, a_b)
        # s = utils.observe(s_r, s_b)
        p1.s = p1.step(a_r)
        bo.s = bo.step(a_b)
        p1.s_train = p1.observe(bo.s)
        list_0.append([p1.s[0], p1.s[1]])
        list_1.append([bo.s[0], bo.s[1]])
        if bo.s[0] < 0 or bo.s[0] > my_utils.MAX_X or bo.s[1] < 0 or bo.s[1] > my_utils.MAX_Y:
            p1.r = 1
        elif p1.s[0] < 0 or p1.s[0] > my_utils.MAX_X or p1.s[1] < 0 or p1.s[1] > my_utils.MAX_Y:
            p1.r = -1
        else:
            # r = utils.reward(s)
            p1.reward_test()
        # print(p1.r)
        # print(p1.s)
        # print(bo.s)
        # print(a_r)
        # print(a_b)
        if p1.r != 0:
            break
    return list_0, list_1, p1.r


def draw(list_0, list_1, r):
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
    plt.figure(1)
    plt.scatter(x_0[0], y_0[0])
    plt.scatter(x_1[0], y_1[0])
    plt.plot(x_0, y_0, 'r')
    plt.plot(x_1, y_1, 'b')
    if r == 1:
        plt.title('Red wins')
    elif r == -1:
        plt.title('Blue wins')
    else:
        plt.title('Draw')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(0, my_utils.MAX_X)
    plt.ylim(0, my_utils.MAX_Y)
    plt.show()


if __name__ == "__main__":
    sum_r = 0

    actor = my_ddpg.Actor(s_dim=3, a_dim=2).to(my_ddpg.device)
    actor.load_state_dict(torch.load('actor.pth'))
    actor.eval()
    for i in range(100):
        list_0, list_1, r = dd_test(actor)
        if r==1:
            sum_r += 1
        draw(list_0, list_1, r)
    print(sum_r)
    # draw(list_0, list_1, r)
