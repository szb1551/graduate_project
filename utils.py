import pygame



w,h = 500, 350
pygame.init()


class raw_env(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.width = 500
        self.length = 350
        self.archer1_image = pygame.image.load('resource/players.png')
        self.boss_image = pygame.image.load('resource/enemy.png')
        self.bg = pygame.image.load('resource/background.png')
        self.rect = self.archer1_image.get_rect()
        self.rect.left, self.rect.top = 100, 175

    def reset(self):
        pass

    def observe(self):
        pass

    def step(self):
        pass

    def evalute(self):
        pass

    def reward(self):
        pygame.init()
        pygame.display.set_mode((self.width, self.length))
        screen = pygame.display.get_surface()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.display.quit()
                    quit()
                if event.type == pygame.KEYDOWNN:
                    if event.key == pygame.k_w:
                        self.rect.top += 3
                    if event.ket == pygame.k_a:
                        self.rect.left -= 3
                    if event.key == pygame.k_s:
                        self.rect.top -= 3
                    if event.ket == pygame.k_d:
                        self.rect.left += 3


