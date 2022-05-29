import pygame
import os

# 得到当前工程目录
current_dir = os.path.split(os.path.realpath(__file__))[0]
print(current_dir)
# 得到文件名
player1_file = r"%s\tank3.gif" % (current_dir)
player2_file = r"%s\tank4.gif" % (current_dir)
boss_file = r"%s\boss_red2.png" % (current_dir)
background_file = r"%s\background.png" % (current_dir)


def load_player1_file():
    player1 = pygame.image.load(player1_file)
    player1 = pygame.transform.scale(player1,(20,20)).convert_alpha()
    return player1


def load_player2_file():
    player2 = pygame.image.load(player2_file).convert_alpha()
    player2 = pygame.transform.scale(player2, (20, 20)).convert_alpha()
    return player2


def load_boss_file():
    boss = pygame.image.load(boss_file).convert_alpha()
    boss = pygame.transform.scale(boss, (20, 20)).convert_alpha()
    return boss


def load_background_file():
    background = pygame.image.load(background_file).convert()
    return background
