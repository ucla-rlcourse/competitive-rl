import pygame


def pygame_draw(screen, color, path):
    pygame.draw.polygon(screen, color, path)


def vertical_ind(screen, x, y, width, height, value, color):
    pygame.draw.rect(screen, color, (x, y, width, height * value))


def horiz_ind(screen, x, y, width, height, value, color):
    pygame.draw.rect(screen, color, (x, y, width * value, height))


def draw_text(screen, text, x, y, font, color=(255, 255, 255)):
    textsurface = font.render(text, False, color)
    screen.blit(textsurface, (x, y))


def draw_dot(screen, color, center, radius):
    pygame.draw.circle(screen, color, center, radius)
