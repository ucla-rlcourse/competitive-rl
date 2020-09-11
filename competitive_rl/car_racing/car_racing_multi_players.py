"""
Easiest continuous control task to learn from pixels, a top-down racing
environment.
Discrete control is reasonable in this environment as well, on/off
discretization is fine.

State consists of STATE_W x STATE_H pixels.

The reward is -0.1 every frame and +1000/N for every track tile visited, where
N is the total number of tiles visited in the track. For example, if you have
finished in 732 frames, your reward is 1000 - 0.1*732 = 926.8 points.

The game is solved when the agent consistently gets 900+ points. The generated
track is random every episode.

The episode finishes when all the tiles are visited. The car also can go
outside of the PLAYFIELD -  that is far off the track, then it will get -100
and die.

Some indicators are shown at the bottom of the window along with the state RGB
buffer. From left to right: the true speed, four ABS sensors, the steering
wheel position and gyroscope.

To play yourself (it's rather fast for humans), type:

python gym/envs/box2d/car_racing.py

Remember it's a powerful rear-wheel drive car -  don't press the accelerator
and turn at the same time.

Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.
"""
import datetime
import json
import math
import sys
import time
import matplotlib.pyplot as plt
import numpy as np

import Box2D
import pygame
from Box2D.b2 import fixtureDef
from Box2D.b2 import polygonShape
from Box2D.b2 import contactListener

import gym
from gym import spaces


from gym.utils import seeding, EzPickle

#import pyglet
#from pyglet import gl
from competitive_rl.car_racing.car_dynamics import Car
from competitive_rl.car_racing.controller import key_phrase
from competitive_rl.car_racing.pygame_rendering import vertical_ind, horiz_ind, draw_text

STATE_W = 96   # less than Atari 160x192
STATE_H = 96
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800

SCALE = 6.0             # Track scale
TRACK_RAD = 900/SCALE   # Track is heavily morphed circle with this radius
PLAYFIELD = 2000/SCALE  # Game over boundary
FPS = 50                # Frames per second
ZOOM = 2.7              # Camera zoom
ZOOM_FOLLOW = True      # Set to False for fixed view (don't use zoom)


TRACK_DETAIL_STEP = 21/SCALE
TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 40/SCALE
BORDER = 8/SCALE
BORDER_MIN_COUNT = 4

ROAD_COLOR = [0.4, 0.4, 0.4]

window_size = WIN_W, WIN_H = 1000, 800
white = 255, 255, 255
initial_camera_scale = 3
initial_camera_offset = (0, 0)
initial_camera_angle = 0
car_scale = 5


class FrictionDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        self._contact(contact, True)

    def EndContact(self, contact):
        self._contact(contact, False)

    def _contact(self, contact, begin):
        tile = None
        obj = None
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData
        car_number = -1
        if u1 and "road_friction" in u1.__dict__:
            tile = u1
            obj = u2
        if u2 and "road_friction" in u2.__dict__:
            tile = u2
            obj = u1
        if not tile:
            return

        tile.color[0] = ROAD_COLOR[0]
        tile.color[1] = ROAD_COLOR[1]
        tile.color[2] = ROAD_COLOR[2]
        if not obj or "tiles" not in obj.__dict__:
            return
        if "car_number" in obj.__dict__:
            car_number = obj.car_number
        if begin:
            obj.tiles.add(tile)
            if not tile.road_visited[car_number]:
                tile.road_visited[car_number] = True
                self.env.rewards[car_number] += 1000.0/len(self.env.track)
                self.env.tile_visited_count[car_number] += 1
        else:
            obj.tiles.remove(tile)


class CarRacing(gym.Env, EzPickle):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'state_pixels'],
        'video.frames_per_second': FPS
    }

    def __init__(self, num_player=1, verbose=1):
        EzPickle.__init__(self)
        self.seed()
        self.contactListener_keepref = FrictionDetector(self)
        self.world = Box2D.b2World(
            (0, 0),
            contactListener=self.contactListener_keepref
        )
        self.viewer = None
        self.screen = None
        self.invisible_state_window = None
        self.invisible_video_window = None
        self.road = None
        self.cars = []
        self.rewards = []
        self.prev_rewards = []
        self.verbose = verbose
        self.num_player = num_player
        self.track = None
        self.done = []
        self.state = []

        self.camera_offset = initial_camera_offset
        self.camera_scale = initial_camera_scale
        self.camera_no_follow_scale = initial_camera_scale
        self.car_scale = car_scale
        self.camera_angle = initial_camera_angle
        self.camera_follow = -1

        self.show_all_car_obs = True

        self.fd_tile = fixtureDef(
            shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)])
        )

        self.action_space = spaces.Box(np.array([-1, 0, 0]),
                                       np.array([+1, +1, +1]),
                                       dtype=np.float32)  # steer, gas, brake

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(STATE_H, STATE_W, 3),
            dtype=np.uint8
        )

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.road:
            return
        for t in self.road:
            self.world.DestroyBody(t)
        self.road = []
        self.car.destroy()

    def _create_track(self, use_local_track="", record_track_to=""):
        CHECKPOINTS = 12

        # Create checkpoints
        checkpoints = []
        for c in range(CHECKPOINTS):
            noise = self.np_random.uniform(0, 2 * math.pi * 1 / CHECKPOINTS)
            alpha = 2 * math.pi * c / CHECKPOINTS + noise
            rad = self.np_random.uniform(TRACK_RAD / 3, TRACK_RAD)

            if c == 0:
                alpha = 0
                rad = 1.5 * TRACK_RAD
            if c == CHECKPOINTS - 1:
                alpha = 2 * math.pi * c / CHECKPOINTS
                self.start_alpha = 2 * math.pi * (-0.5) / CHECKPOINTS
                rad = 1.5 * TRACK_RAD

            checkpoints.append(
                (alpha, rad * math.cos(alpha), rad * math.sin(alpha))
            )
        self.road = []

        # Go from one checkpoint to another to create track
        x, y, beta = 1.5 * TRACK_RAD, 0, 0
        dest_i = 0
        laps = 0
        track = []
        no_freeze = 2500
        visited_other_side = False
        while True:
            alpha = math.atan2(y, x)
            if visited_other_side and alpha > 0:
                laps += 1
                visited_other_side = False
            if alpha < 0:
                visited_other_side = True
                alpha += 2 * math.pi

            while True:  # Find destination from checkpoints
                failed = True

                while True:
                    dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(
                        checkpoints)]
                    if alpha <= dest_alpha:
                        failed = False
                        break
                    dest_i += 1
                    if dest_i % len(checkpoints) == 0:
                        break

                if not failed:
                    break

                alpha -= 2*math.pi
                continue

            r1x = math.cos(beta)
            r1y = math.sin(beta)
            p1x = -r1y
            p1y = r1x
            dest_dx = dest_x - x  # vector towards destination
            dest_dy = dest_y - y
            # destination vector projected on rad:
            proj = r1x * dest_dx + r1y * dest_dy
            while beta - alpha > 1.5 * math.pi:
                beta -= 2 * math.pi
            while beta - alpha < -1.5 * math.pi:
                beta += 2 * math.pi
            prev_beta = beta
            proj *= SCALE
            if proj > 0.3:
                beta -= min(TRACK_TURN_RATE, abs(0.001 * proj))
            if proj < -0.3:
                beta += min(TRACK_TURN_RATE, abs(0.001 * proj))
            x += p1x * TRACK_DETAIL_STEP
            y += p1y * TRACK_DETAIL_STEP
            track.append((alpha, prev_beta * 0.5 + beta * 0.5, x, y))
            if laps > 4:
                break
            no_freeze -= 1
            if no_freeze == 0:
                break

        # Find closed loop range i1..i2, first loop should be ignored, second is OK
        i1, i2 = -1, -1
        i = len(track)
        while True:
            i -= 1
            if i == 0:
                return False  # Failed
            pass_through_start = track[i][0] > self.start_alpha and track[i -
                                                                          1][0] <= self.start_alpha
            if pass_through_start and i2 == -1:
                i2 = i
            elif pass_through_start and i1 == -1:
                i1 = i
                break
        if self.verbose == 1:
            print("Track generation: %i..%i -> %i-tiles track" %
                  (i1, i2, i2-i1))
        assert i1 != -1
        assert i2 != -1

        track = track[i1:i2-1]

        first_beta = track[0][1]
        first_perp_x = math.cos(first_beta)
        first_perp_y = math.sin(first_beta)
        # Length of perpendicular jump to put together head and tail
        well_glued_together = np.sqrt(
            np.square(first_perp_x * (track[0][2] - track[-1][2])) +
            np.square(first_perp_y * (track[0][3] - track[-1][3])))
        if well_glued_together > TRACK_DETAIL_STEP:
            return False

        if use_local_track != "":
            track = []
            file = open(use_local_track, 'r', encoding='utf-8')
            data = json.load(file)
            for i in range(len(data)):
                track.append((data[i][0], data[i][1], data[i][2], data[i][3]))
            print(track)
        # Red-white border on hard turns
        border = [False] * len(track)
        for i in range(len(track)):
            good = True
            oneside = 0
            for neg in range(BORDER_MIN_COUNT):
                beta1 = track[i-neg-0][1]
                beta2 = track[i-neg-1][1]
                good &= abs(beta1 - beta2) > TRACK_TURN_RATE * 0.2
                oneside += np.sign(beta1 - beta2)
            good &= abs(oneside) == BORDER_MIN_COUNT
            border[i] = good
        for i in range(len(track)):
            for neg in range(BORDER_MIN_COUNT):
                border[i-neg] |= border[i]

        # Create tiles
        for i in range(len(track)):
            alpha1, beta1, x1, y1 = track[i]
            alpha2, beta2, x2, y2 = track[i-1]
            road1_l = (x1 - TRACK_WIDTH*math.cos(beta1),
                       y1 - TRACK_WIDTH*math.sin(beta1))
            road1_r = (x1 + TRACK_WIDTH*math.cos(beta1),
                       y1 + TRACK_WIDTH*math.sin(beta1))
            road2_l = (x2 - TRACK_WIDTH*math.cos(beta2),
                       y2 - TRACK_WIDTH*math.sin(beta2))
            road2_r = (x2 + TRACK_WIDTH*math.cos(beta2),
                       y2 + TRACK_WIDTH*math.sin(beta2))
            vertices = [road1_l, road1_r, road2_r, road2_l]
            self.fd_tile.shape.vertices = vertices
            t = self.world.CreateStaticBody(fixtures=self.fd_tile)
            t.userData = t
            c = 0.01*(i % 3)
            t.color = [ROAD_COLOR[0] + c, ROAD_COLOR[1] + c, ROAD_COLOR[2] + c]
            t.road_visited = [False] * self.num_player
            t.road_friction = 1.0
            t.fixtures[0].sensor = True
            self.road_poly.append(
                ([road1_l, road1_r, road2_r, road2_l], t.color)
            )
            self.road.append(t)
            if border[i]:
                side = np.sign(beta2 - beta1)
                b1_l = (x1 + side * TRACK_WIDTH * math.cos(beta1),
                        y1 + side * TRACK_WIDTH * math.sin(beta1))
                b1_r = (x1 + side * (TRACK_WIDTH+BORDER) * math.cos(beta1),
                        y1 + side * (TRACK_WIDTH+BORDER)*math.sin(beta1))
                b2_l = (x2 + side * TRACK_WIDTH * math.cos(beta2),
                        y2 + side * TRACK_WIDTH * math.sin(beta2))
                b2_r = (x2 + side * (TRACK_WIDTH+BORDER) * math.cos(beta2),
                        y2 + side * (TRACK_WIDTH+BORDER) * math.sin(beta2))
                self.road_poly.append(
                    (
                        [b1_l, b1_r, b2_r, b2_l],
                        (1, 1, 1) if i % 2 == 0 else (1, 0, 0)
                    )
                )
        self.track = track
        if record_track_to != "":
            file = open(record_track_to + "/" + datetime.datetime.fromtimestamp(time.time()).strftime(
                f"%Y-%m-%d_%H-%M-%S_track.json"
            ), 'w', encoding='utf-8')
            json.dump(track, file)
        return True

    def reset(self, use_local_track="", record_track_to=""):
        self._destroy()
        self.rewards = [0] * self.num_player
        self.prev_rewards = [0] * self.num_player
        self.tile_visited_count = [0] * self.num_player
        self.done = [0]*self.num_player
        self.t = 0.0
        self.road_poly = []
        if use_local_track != "":
            self._create_track(use_local_track=use_local_track,
                               record_track_to=record_track_to)
        else:
            while True:
                success = self._create_track(
                    use_local_track=use_local_track, record_track_to=record_track_to)
                if success:
                    break
                if self.verbose == 1:
                    print(
                        "retry to generate track (normal if there are not many"
                        "instances of this message)"
                    )
        for i in range(self.num_player):
            self.cars.append(Car(self.world, *self.track[0][1:4], i))

        return self.step(None)[0]

    # Return [observation, reward, done, info]
    def step(self, action):
        if action is not None:
            for i in range(len(self.cars)):
                self.cars[i].steer(-action[i][0])
                self.cars[i].gas(action[i][1])
                self.cars[i].brake(action[i][2])
        for car in self.cars:
            car.step(1.0/FPS)
        self.world.Step(1.0/FPS, 6*30, 2*30)
        self.t += 1.0/FPS

        step_rewards = [0] * self.num_player
        '''if action is not None:  # First step without action, called from reset()
            self.reward -= 0.1
            # We actually don't want to count fuel spent, we want car to be faster.
            # self.reward -=  10 * self.car.fuel_spent / ENGINE_POWER
            self.car.fuel_spent = 0.0
            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward
            if self.tile_visited_count == len(self.track):
                done = True
            x, y = self.car.hull.position
            if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                done = True
                step_reward = -100'''

        if action is not None:  # First step without action, called from reset()
            for i in range(self.num_player):
                self.rewards[i] -= 0.1
                # We actually don't want to count fuel spent, we want car to be faster.
                # self.reward -=  10 * self.car.fuel_spent / ENGINE_POWER
                self.cars[i].fuel_spent = 0.0
                step_rewards[i] = self.rewards[i] - self.prev_rewards[i]
                self.prev_rewards[i] = self.rewards[i]
                x, y = self.cars[i].hull.position
                if self.tile_visited_count[i] == len(self.track):
                    self.done[i] = 1
                if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                    self.done[i] = 1
                    step_rewards[i] = -100

        return self.state,step_rewards, self.done, {}

    '''def render(self, mode='human'):
        assert mode in ['human', 'state_pixels', 'rgb_array']
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
            self.score_label = pyglet.text.Label(
                '0000',
                font_size=36,
                x=20,
                y=WINDOW_H * 2.5 / 40.00,
                anchor_x='left',
                anchor_y='center',
                color=(255, 255, 255, 255)
            )
            self.transform = rendering.Transform()

        if "t" not in self.__dict__:
            return  # reset() not called yet

        # Animate zoom first second:
        #zoom = 0.1 * SCALE * max(1 - self.t, 0) + ZOOM * SCALE * min(self.t, 1)
        zoom = 8
        scroll_x = self.cars[0].hull.position[0]
        scroll_y = self.cars[0].hull.position[1]
        angle = -self.cars[0].hull.angle
        vel = self.cars[0].hull.linearVelocity
        #if np.linalg.norm(vel) > 0.5:
            #angle = math.atan2(vel[0], vel[1])
        self.transform.set_scale(zoom, zoom)
        self.transform.set_translation(
            WINDOW_W/2
            - (scroll_x * zoom * math.cos(angle) - scroll_y * zoom * math.sin(angle)),
            WINDOW_H/2
            - (scroll_x * zoom * math.sin(angle) + scroll_y * zoom * math.cos(angle))
        )
        self.transform.set_rotation(angle)

        for car in self.cars:
            car.draw(self.viewer, mode != "state_pixels")

        arr = None
        win = self.viewer.window
        win.switch_to()
        win.dispatch_events()

        win.clear()
        t = self.transform
        if mode == 'rgb_array':
            VP_W = VIDEO_W
            VP_H = VIDEO_H
        elif mode == 'state_pixels':
            VP_W = STATE_W
            VP_H = STATE_H
        else:
            pixel_scale = 1
            if hasattr(win.context, '_nscontext'):
                pixel_scale = win.context._nscontext.view().backingScaleFactor()  # pylint: disable=protected-access
            VP_W = int(pixel_scale * WINDOW_W)
            VP_H = int(pixel_scale * WINDOW_H)

        #gl.glViewport(0, 0, VP_W, VP_H)
        t.enable()
        self.render_road()
        for geom in self.viewer.onetime_geoms:
            geom.render()
        self.viewer.onetime_geoms = []
        t.disable()
        #self.render_indicators(WINDOW_W, WINDOW_H)

        if mode == 'human':
            win.flip()
            return self.viewer.isopen

        #image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        #arr = np.fromstring(image_data.get_data(), dtype=np.uint8, sep='')
        #arr = arr.reshape(VP_H, VP_W, 4)
        #arr = arr[::-1, :, 0:3]

        return arr'''

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    '''def render_road(self):
        gl.glBegin(gl.GL_QUADS)
        gl.glColor4f(0.4, 0.8, 0.4, 1.0)
        gl.glVertex3f(-PLAYFIELD, +PLAYFIELD, 0)
        gl.glVertex3f(+PLAYFIELD, +PLAYFIELD, 0)
        gl.glVertex3f(+PLAYFIELD, -PLAYFIELD, 0)
        gl.glVertex3f(-PLAYFIELD, -PLAYFIELD, 0)
        gl.glColor4f(0.4, 0.9, 0.4, 1.0)
        k = PLAYFIELD/20.0
        for x in range(-20, 20, 2):
            for y in range(-20, 20, 2):
                gl.glVertex3f(k * x + k, k * y + 0, 0)
                gl.glVertex3f(k * x + 0, k * y + 0, 0)
                gl.glVertex3f(k * x + 0, k * y + k, 0)
                gl.glVertex3f(k * x + k, k * y + k, 0)
        for poly, color in self.road_poly:
            gl.glColor4f(color[0], color[1], color[2], 1)
            for p in poly:
                gl.glVertex3f(p[0], p[1], 1)
        gl.glEnd()'''

    '''def render_indicators(self, W, H):
        gl.glBegin(gl.GL_QUADS)
        s = W / 40.0
        h = H / 40.0
        gl.glColor4f(0, 0, 0, 1)
        gl.glVertex3f(W, 0, 0)
        gl.glVertex3f(W, 5 * h, 0)
        gl.glVertex3f(0, 5 * h, 0)
        gl.glVertex3f(0, 0, 0)

        def vertical_ind(place, val, color):
            gl.glColor4f(color[0], color[1], color[2], 1)
            gl.glVertex3f((place+0) * s, h + h * val, 0)
            gl.glVertex3f((place+1) * s, h + h * val, 0)
            gl.glVertex3f((place+1) * s, h, 0)
            gl.glVertex3f((place+0) * s, h, 0)

        def horiz_ind(place, val, color):
            gl.glColor4f(color[0], color[1], color[2], 1)
            gl.glVertex3f((place + 0) * s, 4 * h, 0)
            gl.glVertex3f((place + val) * s, 4 * h, 0)
            gl.glVertex3f((place + val) * s, 2 * h, 0)
            gl.glVertex3f((place + 0) * s, 2 * h, 0)
        true_speed = np.sqrt(
            np.square(self.car.hull.linearVelocity[0])
            + np.square(self.car.hull.linearVelocity[1])
        )
        vertical_ind(5, 0.02*true_speed, (1, 1, 1))
        vertical_ind(7, 0.01*self.car.wheels[0].omega, (0.0, 0, 1))  # ABS sensors
        vertical_ind(8, 0.01*self.car.wheels[1].omega, (0.0, 0, 1))
        vertical_ind(9, 0.01*self.car.wheels[2].omega, (0.2, 0, 1))
        vertical_ind(10, 0.01*self.car.wheels[3].omega, (0.2, 0, 1))
        horiz_ind(20, -10.0 * self.car.wheels[0].joint.angle, (0, 1, 0))
        horiz_ind(30, -0.8 * self.car.hull.angularVelocity, (1, 0, 0))
        gl.glEnd()
        self.score_label.text = "%04i" % self.reward
        self.score_label.draw()'''

    def render_indicators_for_pygame(self, screen, W=WIN_W, H=WIN_H):
        if self.camera_follow < 0:
            return
        s = W / 40.0
        h = H / 40.0

        true_speed = np.sqrt(
            np.square(self.cars[self.camera_follow].hull.linearVelocity[0])
            + np.square(self.cars[self.camera_follow].hull.linearVelocity[1])
        )
        vertical_ind(screen, 5 * s, h, s, h, 0.02*true_speed, (0, 0, 255))

        vertical_ind(screen, 7 * s, h, s, h, 0.01 *
                     self.cars[self.camera_follow].wheels[0].omega, (0.0, 0, 255))  # ABS sensors
        vertical_ind(screen, 8 * s, h, s, h, 0.01 *
                     self.cars[self.camera_follow].wheels[1].omega, (0.0, 0, 255))
        vertical_ind(screen, 9 * s, h, s, h, 0.01 *
                     self.cars[self.camera_follow].wheels[2].omega, (0.2 * 255, 0, 255))
        vertical_ind(screen, 10 * s, h, s, h, 0.01 *
                     self.cars[self.camera_follow].wheels[3].omega, (0.2 * 255, 0, 1))
        horiz_ind(screen, 20 * s, 2 * h, s, 2 * h, -10.0 *
                  self.cars[self.camera_follow].wheels[0].joint.angle, (0, 255, 0))
        horiz_ind(screen, 30 * s, 2 * h, s, 2 * h, -0.8 *
                  self.cars[self.camera_follow].hull.angularVelocity, (255, 0, 0))
        draw_text(screen, str(self.rewards[self.camera_follow]), 10, 10)

    def render_road_for_pygame(self, screen):
        screen.fill((0.4 * 255, 0.8 * 255, 0.4 * 255))
        for poly, color in self.road_poly:
            tmp = Box2D.b2Transform()
            tmp.position = (0, 0)
            tmp.angle = -self.camera_angle
            # trans = Box2D.b2Transform()
            path_tmp = [(-v[0] + self.camera_offset[0], -v[1] +
                         self.camera_offset[1]) for v in poly]
            path = [self.camera_scale*(tmp*v) + (WIN_W/2, WIN_H/2)
                    for v in path_tmp]
            #self.object_to_draw.append(([255 * i for i in color], path))
            pygame.draw.polygon(screen, [255 * i for i in color], path)

    def camera_update(self):
        if (self.camera_follow != -1):
            self.camera_offset = self.cars[self.camera_follow].hull.position
            self.camera_angle = self.cars[self.camera_follow].hull.angle
            self.camera_scale = self.car_scale
        else:
            self.camera_offset = initial_camera_offset
            self.camera_angle = initial_camera_angle
            self.camera_scale = initial_camera_scale

    def manage_input(self, input_number):
        if input_number != None:
            if input_number >= -1:
                self.camera_follow = input_number
            if input_number == -2:
                self.show_all_car_obs = True

    def get_rendered_screen(self, surface=None):
        if surface == None:
            surface = pygame.Surface(window_size)
        env.render_road_for_pygame(surface)

        for car in env.cars:
            car.draw_for_pygame(surface, WIN_W, WIN_H, offset=env.camera_offset, angle=env.camera_angle,
                                scale=env.camera_scale)

        env.render_indicators_for_pygame(surface)

        # Update state of each car and append it to state array
        original_follow = self.camera_follow
        obs_arr = []
        for i in range(self.num_player):
            self.camera_follow = i
            self.camera_update()
            obs_top_left_x = (WIN_W - STATE_W) / 2
            obs_top_left_y = (WIN_H - STATE_H) / 2
            obs_surface = surface.subsurface(obs_top_left_x, obs_top_left_y, STATE_W, STATE_H)
            obs_arr.append(pygame.surfarray.array3d(obs_surface))
        self.state = obs_arr
        
        # if (self.rewards[0] + 1) > 200:
        #     plt.imshow(self.state[0])
        #     plt.show()
        
        # Return camera to normal position
        self.camera_follow = original_follow
        self.camera_update()
        
        return surface

if __name__ == "__main__":
    pygame.init()
    pygame.font.init()
    env = CarRacing(num_player=2)

    env.reset(use_local_track="", record_track_to="")
    a = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    screen = pygame.display.set_mode(window_size, pygame.RESIZABLE)
    while True:
        screen.fill(white)
        env.manage_input(key_phrase(a))
        env.camera_update()
        observation, reward, done, info = env.step(a)

        if(len(observation)):
            print(observation[0].shape)

        screen = env.get_rendered_screen(screen)

 
        pygame.display.update()
