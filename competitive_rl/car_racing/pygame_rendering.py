import pygame
from Box2D import Box2D

from gym.envs.box2d.car_dynamics import Car
from gym.envs.box2d.car_racing_multi_players import CarRacing
from gym.envs.classic_control.rendering import Transform, gl

if __name__ == "__main__":
    import sys, pygame


    pygame.init()
    size = width, height = 1920,1080
    speed = [2, 2]
    white = 255, 255, 255
    screen = pygame.display.set_mode(size, pygame.RESIZABLE)
    fake_screen = screen.copy()

    #car = Car(world, 1, 10, 10)
    playground = pygame.surface.Surface((10, 10))
    playground.fill(white)
    screen.fill(white)

    from pyglet.window import key

    a = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    ''''''
    def key_press(k, mod):
        global restart
        if k == 0xff0d: restart = True
        if k == key.LEFT:  a[0][0] = -1.0
        if k == key.RIGHT: a[0][0] = +1.0
        if k == key.UP:    a[0][1] = +1.0
        if k == key.DOWN:  a[0][2] = +0.8  # set 1.0 for wheels to block to zero rotation
        if k == key.A:  a[1][0] = -1.0
        if k == key.D: a[1][0] = +1.0
        if k == key.W:    a[1][1] = +1.0
        if k == key.S:  a[1][2] = +0.8  # set 1.0 for wheels to block to zero rotation


    def key_release(k, mod):
        if k == key.LEFT and a[0][0] == -1.0: a[0][0] = 0
        if k == key.RIGHT and a[0][0] == +1.0: a[0][0] = 0
        if k == key.UP:    a[0][1] = 0
        if k == key.DOWN:  a[0][2] = 0
        if k == key.A and a[1][0] == -1.0: a[1][0] = 0
        if k == key.D and a[1][0] == +1.0: a[1][0] = 0
        if k == key.W:    a[1][1] = 0
        if k == key.S:  a[1][2] = 0


    env = CarRacing(num_player=2)
    #env.render()
    #env.viewer.window.on_key_press = key_press
    #env.viewer.window.on_key_release = key_release
    env.reset()
    isopen = True
    '''while isopen:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        env.cars[0].draw_for_pygame(playground)
        fake_screen.blit(playground, (0, 0))
        pygame.display.update()
        while True:
            s, r, done, info = env.step(a)
            steps += 1
            isopen = env.render()
            env.cars[0].draw_for_pygame(playground)
            fake_screen.blit(playground, (0, 0))
            pygame.display.update()
            print("h")
    env.close()'''

    ''''''
    #offset = size
    initial_scale = 3
    camera_offset = (0,0)
    camera_scale = initial_scale
    car_scale = 5
    camera_angle = 0
    camera_follow = -1
    while True:
        screen.fill(white)
        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_0:
                    camera_follow = -1
                    camera_offset = (0,0)
                    camera_scale = initial_scale
                    camera_angle = 0
                elif event.key == pygame.K_1:
                    camera_follow = 0
                    camera_offset = env.cars[0].hull.position
                    camera_angle = env.cars[0].hull.angle
                    camera_scale = car_scale
                elif event.key == pygame.K_2:
                    camera_follow = 1
                    camera_offset = env.cars[1].hull.position
                    camera_angle = env.cars[1].hull.angle
                    camera_scale = car_scale
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
        #env.cars[0].draw_for_pygame(playground)
        #fake_screen.blit(playground, (0, 0))
        #screen.blit(pygame.transform.rotozoom(fake_screen, 0, 10), (0,0))
        env.step(a)
        if(camera_follow != -1):
            camera_offset = env.cars[camera_follow].hull.position
            camera_angle = env.cars[camera_follow].hull.angle
            camera_scale = car_scale
        env.render_road_for_pygame(screen, width, height, offset=camera_offset, angle=camera_angle, scale=camera_scale)
        env.cars[0].draw_for_pygame(screen, width, height, offset=camera_offset, angle=camera_angle, scale=camera_scale)
        env.cars[1].draw_for_pygame(screen, width, height, offset=camera_offset, angle=camera_angle, scale=camera_scale)
        pygame.display.update()
