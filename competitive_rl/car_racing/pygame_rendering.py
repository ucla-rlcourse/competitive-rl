import pygame

def pygame_draw(screen, color, path):
    pygame.draw.polygon(screen, color, path)


def vertical_ind(screen, x, y, width, height, value, color):
    pygame.draw.rect(screen, color, (x, y, width, height * value))


def horiz_ind(screen, x, y, width, height, value, color):
    pygame.draw.rect(screen, color, (x, y, width * value, height))


