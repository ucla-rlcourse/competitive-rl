import sys

import pygame

def key_phrase(a):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_z:
                return -2
            elif event.key == pygame.K_r:
                return -4
            elif event.key == pygame.K_0:
                return -1
            elif event.key == pygame.K_1:
                return 0
            elif event.key == pygame.K_2:
                return 1
            elif event.key == pygame.K_3:
                return 2
            elif event.key == pygame.K_4:
                return 3
            elif event.key == pygame.K_a:
                a[1][0] = +1.0
            elif event.key == pygame.K_d:
                a[1][0] = -1.0
            elif event.key == pygame.K_w:
                a[1][1] = +1.0
            elif event.key == pygame.K_s:
                a[1][2] = +0.8
            elif event.key == pygame.K_LEFT:
                a[0][0] = +1.0
            elif event.key == pygame.K_RIGHT:
                a[0][0] = -1.0
            elif event.key == pygame.K_UP:
                a[0][1] = +1.0
            elif event.key == pygame.K_DOWN:
                a[0][2] = +0.8
            elif event.key == pygame.K_f:
                a[2][0] = +1.0
            elif event.key == pygame.K_h:
                a[2][0] = -1.0
            elif event.key == pygame.K_t:
                a[2][1] = +1.0
            elif event.key == pygame.K_g:
                a[2][2] = +0.8
            elif event.key == pygame.K_j:
                a[3][0] = +1.0
            elif event.key == pygame.K_l:
                a[3][0] = -1.0
            elif event.key == pygame.K_i:
                a[3][1] = +1.0
            elif event.key == pygame.K_k:
                a[3][2] = +0.8
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_a and a[1][0] == +1.0:
                a[1][0] = 0.0
            elif event.key == pygame.K_d and a[1][0] == -1.0:
                a[1][0] = +0.0
            elif event.key == pygame.K_w:
                a[1][1] = +0.0
            elif event.key == pygame.K_s:
                a[1][2] = +0.0
            elif event.key == pygame.K_LEFT and a[0][0] == +1.0:
                a[0][0] = 0.0
            elif event.key == pygame.K_RIGHT and a[0][0] == -1.0:
                a[0][0] = +0.0
            elif event.key == pygame.K_UP:
                a[0][1] = +0.0
            elif event.key == pygame.K_DOWN:
                a[0][2] = +0.0
            elif event.key == pygame.K_f and a[2][0] == +1.0:
                a[2][0] = 0.0
            elif event.key == pygame.K_h and a[2][0] == -1.0:
                a[2][0] = +0.0
            elif event.key == pygame.K_t:
                a[2][1] = +0.0
            elif event.key == pygame.K_g:
                a[2][2] = +0.0
            elif event.key == pygame.K_j and a[3][0] == +1.0:
                a[3][0] = 0.0
            elif event.key == pygame.K_l and a[3][0] == -1.0:
                a[3][0] = +0.0
            elif event.key == pygame.K_i:
                a[3][1] = +0.0
            elif event.key == pygame.K_k:
                a[3][2] = +0.0
