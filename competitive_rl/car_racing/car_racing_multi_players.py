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
import time

import Box2D
import gym
import matplotlib.pyplot as plt
import numpy as np
import pygame
from Box2D.b2 import fixtureDef, polygonShape, contactListener
from gym import spaces
from gym.utils import seeding, EzPickle

from competitive_rl.car_racing.car_dynamics import Car
from competitive_rl.car_racing.pygame_rendering import vertical_ind, horiz_ind, draw_text

pygame.init()
pygame.font.init()

STATE_W = 96  # less than Atari 160x192
STATE_H = 96
# VIDEO_W = 600
# VIDEO_H = 400
# WINDOW_W = 1000
# WINDOW_H = 800
SCALE = 6.0  # Track scale
TRACK_RAD = 900 / SCALE  # Track is heavily morphed circle with this radius
PLAYFIELD = 2000 / SCALE  # Game over boundary
FPS = 50  # Frames per second
ZOOM = 2.7  # Camera zoom
ZOOM_FOLLOW = True  # Set to False for fixed view (don't use zoom)

TRACK_DETAIL_STEP = 21 / SCALE
TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 40 / SCALE
BORDER = 8 / SCALE
BORDER_MIN_COUNT = 4

ROAD_COLOR = [0.4, 0.4, 0.4]

WINDOW_SIZE = width, height = 1000, 800
white = 255, 255, 255
initial_camera_scale = 1
tmp = Box2D.b2Transform()
tmp.position = (0, 0)
initial_camera_offset = tmp
initial_camera_angle = 0
car_scale = 15
num_player = 1


class FrictionDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        self._contact(contact, True)
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData
        if u1 and u2:
            if u1 and "car_number" in u1.__dict__ and u2 and "car_number" in u2.__dict__:
                print("deduced")
                self.env.rewards[u1.car_number] -= 1000 / FPS
                self.env.rewards[u2.car_number] -= 1000 / FPS
                return

    def EndContact(self, contact):
        self._contact(contact, False)

    def _contact(self, contact, begin):
        tile = None
        obj = None
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData

        car_number = -1
        if u1 and "road_friction" in u1.__dict__:
            self.env.ontrack_count = 0
            tile = u1
            obj = u2
        if u2 and "road_friction" in u2.__dict__:
            self.env.ontrack_count = 0
            tile = u2
            obj = u1
        if not tile:
            return
        if "car_number" in obj.__dict__:
            car_number = obj.car_number
        tile.color[0] = ROAD_COLOR[0]
        tile.color[1] = ROAD_COLOR[1]
        tile.color[2] = ROAD_COLOR[2]
        if not obj or "tiles" not in obj.__dict__:
            return

        if begin:
            obj.tiles.add(tile)
            if not tile.road_visited[car_number]:
                tile.road_visited[car_number] = True
                # self.env.rewards[car_number] += 1000.0/len(self.env.track)
                self.env.rewards[car_number] += 10
                self.env.tile_visited_count[car_number] += 1
        else:
            obj.tiles.remove(tile)


class CarRacing(gym.Env, EzPickle):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': FPS
    }

    def __init__(self, num_player=1, verbose=1, seed=8367813160709901366, window_size=WINDOW_SIZE):
        EzPickle.__init__(self)
        self.seed(seed=seed)
        self.contactListener_keepref = FrictionDetector(self)
        self.world = Box2D.b2World(
            (0, 0),
            contactListener=self.contactListener_keepref
        )
        self.viewer = None
        pygame.init()
        pygame.font.init()
        self.window_size = window_size
        self.screen = pygame.display.set_mode(window_size)
        self.playground_surface = pygame.display.set_mode(window_size)
        self.invisible_state_window = None
        self.invisible_video_window = None
        self.road = None
        self.cars = []
        self.rewards = [0] * num_player
        self.prev_rewards = [0] * num_player
        self.verbose = verbose
        self.num_player = num_player
        self.track = None
        self.done = []
        self.background = None
        self.road_poly = None

        self.camera_offset = None
        self.camera_scale = None
        self.camera_no_follow_scale = None
        self.car_scale = None
        self.camera_angle = None
        self.camera_follow = -1

        self.show_all_car_obs = False
        self.ontrack_count = 0
        self.tile_visited_count = None
        self.idle_count = None
        self.step_count = None

        self.isopen = True

        self.world_map = None
        self.world_scale = 10
        self.obs_scale = (self.world_scale / (100 / math.sqrt(96))) * 1.8
        self.world_size = 10000, 10000

        self.obs = {}
        self.info = None

        self.playground = None
        self.observation_playground = None
        self.observation_screens = None

        self.fd_tile = None

        self.action_space = spaces.Box(np.array([-1, 0, 0]),
                                       np.array([+1, +1, +1]),
                                       dtype=np.float32)  # steer, gas, brake

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(STATE_W, STATE_H, 1),
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
        self.cars = []

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
                    dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
                    if alpha <= dest_alpha:
                        failed = False
                        break
                    dest_i += 1
                    if dest_i % len(checkpoints) == 0:
                        break

                if not failed:
                    break

                alpha -= 2 * math.pi
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
            pass_through_start = track[i][0] > self.start_alpha and track[i - 1][0] <= self.start_alpha
            if pass_through_start and i2 == -1:
                i2 = i
            elif pass_through_start and i1 == -1:
                i1 = i
                break
        if self.verbose == 1:
            print("Track generation: %i..%i -> %i-tiles track" % (i1, i2, i2 - i1))
        assert i1 != -1
        assert i2 != -1

        track = track[i1:i2 - 1]

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
            # print(track)
        # Red-white border on hard turns
        border = [False] * len(track)
        for i in range(len(track)):
            good = True
            oneside = 0
            for neg in range(BORDER_MIN_COUNT):
                beta1 = track[i - neg - 0][1]
                beta2 = track[i - neg - 1][1]
                good &= abs(beta1 - beta2) > TRACK_TURN_RATE * 0.2
                oneside += np.sign(beta1 - beta2)
            good &= abs(oneside) == BORDER_MIN_COUNT
            border[i] = good
        for i in range(len(track)):
            for neg in range(BORDER_MIN_COUNT):
                border[i - neg] |= border[i]

        # Create tiles
        for i in range(len(track)):
            alpha1, beta1, x1, y1 = track[i]
            alpha2, beta2, x2, y2 = track[i - 1]
            road1_l = (x1 - TRACK_WIDTH * math.cos(beta1), y1 - TRACK_WIDTH * math.sin(beta1))
            road1_r = (x1 + TRACK_WIDTH * math.cos(beta1), y1 + TRACK_WIDTH * math.sin(beta1))
            road2_l = (x2 - TRACK_WIDTH * math.cos(beta2), y2 - TRACK_WIDTH * math.sin(beta2))
            road2_r = (x2 + TRACK_WIDTH * math.cos(beta2), y2 + TRACK_WIDTH * math.sin(beta2))
            vertices = [road1_l, road1_r, road2_r, road2_l]
            self.fd_tile.shape.vertices = vertices
            t = self.world.CreateStaticBody(fixtures=self.fd_tile)
            t.userData = t
            c = 0.01 * (i % 3)
            t.color = [ROAD_COLOR[0] + c, ROAD_COLOR[1] + c, ROAD_COLOR[2] + c]
            '''t.color = []
            for i in range(self.num_player):
                t.color.append([ROAD_COLOR[0] + c, ROAD_COLOR[1] + c, ROAD_COLOR[2] + c])'''
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
                b1_r = (x1 + side * (TRACK_WIDTH + BORDER) * math.cos(beta1),
                        y1 + side * (TRACK_WIDTH + BORDER) * math.sin(beta1))
                b2_l = (x2 + side * TRACK_WIDTH * math.cos(beta2),
                        y2 + side * TRACK_WIDTH * math.sin(beta2))
                b2_r = (x2 + side * (TRACK_WIDTH + BORDER) * math.cos(beta2),
                        y2 + side * (TRACK_WIDTH + BORDER) * math.sin(beta2))
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
        self.contactListener_keepref = FrictionDetector(self)
        self.world = Box2D.b2World(
            (0, 0),
            contactListener=self.contactListener_keepref
        )
        # Reset camera
        self.camera_offset = initial_camera_offset
        self.camera_scale = initial_camera_scale
        self.camera_no_follow_scale = initial_camera_scale
        self.car_scale = car_scale
        self.camera_angle = initial_camera_angle

        # Reset rewards related
        if self.verbose == 1:
            print("*" * 50)
            print(f"Reset with {self.step_count}steps")
            print(f"Score: {self.rewards}")
            if self.tile_visited_count:
                print(f"Tile visited: {self.tile_visited_count[0]}")
            print("*" * 50)
            print()
        self.rewards = [0] * self.num_player
        self.prev_rewards = [0] * self.num_player
        self.tile_visited_count = [0] * self.num_player
        self.outbound_count = 0
        self.idle_count = 0
        self.step_count = 0

        self.done = [0] * self.num_player
        self.t = 0.0
        self.road_poly = []
        self.fd_tile = fixtureDef(
            shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)])
        )

        # Reset rendering related
        self.observation_screens = [pygame.Surface((STATE_W, STATE_H))] * self.num_player
        self.world_map = pygame.Surface(self.world_size)
        self.playground = pygame.Surface(self.world_size)
        if use_local_track != "":
            self._create_track(use_local_track=use_local_track, record_track_to=record_track_to)
        else:
            while True:
                success = self._create_track(use_local_track=use_local_track, record_track_to=record_track_to)
                if success:
                    break
                if self.verbose == 1:
                    print(
                        "retry to generate track (normal if there are not many"
                        "instances of this message)"
                    )
        for i in range(self.num_player):
            # print(*self.track[0][1:4])
            self.cars.append(Car(self.world, *self.track[0][1:4], i))
        self.playground = self.render_road_for_world_map()
        self.observation_playground = self.render_road_for_observation_map()

        self.info = ["" for x in range(self.num_player)]

        self.camera_follow = 0
        self.camera_update()
        return self.step(None)[0]

    def step(self, action):
        if action is not None:
            action = np.clip(action, -1, 1)
        if pygame.display.list_modes() != -1:  # If not in remote server
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
        if action is not None:
            if self.num_player > 1:
                for i in range(len(self.cars)):
                    self.cars[i].steer(-action[i][0])
                    self.cars[i].gas(action[i][1])
                    self.cars[i].brake(action[i][2])
            if self.num_player == 1:
                self.cars[0].steer(-action[0])
                self.cars[0].gas(action[1])
                self.cars[0].brake(action[2])

        for car in self.cars:
            car.step(1.0 / FPS)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / FPS
        self.ontrack_count += 1
        self.step_count += 1

        step_rewards = [0] * self.num_player
        if action is not None:  # First step without action, called from reset()
            for i in range(self.num_player):
                self.rewards[i] -= 0.1
                # We actually don't want to count fuel spent, we want car to be faster.
                # self.reward -=  10 * self.car.fuel_spent / ENGINE_POWER
                self.cars[i].fuel_spent = 0.0
                step_rewards[i] = self.rewards[i] - self.prev_rewards[i]
                self.prev_rewards[i] = self.rewards[i]
                x, y = self.cars[i].hull.position

                self.info[i] = f"P{i}: Tiles_visited: {self.tile_visited_count[i]} / {len(self.track)}, "
                self.info[i] += f"rewards: {step_rewards[i]}"

                if self.tile_visited_count[i] == len(self.track):
                    # print("car finishs all tiles")
                    self.done[i] = 1
                if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                    # print("car out of playfield")
                    self.done[i] = 1
                # if self.ontrack_count >= 190:
                #     print("Out of Road")
                #     self.done[i] = 1
                # step_rewards[i] = -50
                if self.step_count > 800 and self.tile_visited_count[0] < 2:
                    print("Killed, idle")
                    step_rewards[i] = -100
                    self.done[i] = 1

        # Centralize the logic of rendering observation state into the step function
        original_follow = self.camera_follow
        for i in range(self.num_player):
            self.obs[i] = self.get_observation(i)

        self.camera_follow = original_follow
        self.camera_update()

        # Reward clipping
        step_rewards = np.clip(step_rewards, -1, 1)

        if self.num_player == 1:
            return self.obs[0], step_rewards[0], self.done[0], {}

        return self.obs, step_rewards, self.done, {}

    def get_observation(self, agent_index):
        i = agent_index
        self.camera_follow = i
        self.camera_update("rgb_array")
        self.render(playground=self.observation_playground, playground_surface=self.observation_screens[i],
                    mode="internal_rgb_array", drawing_for_player_num=i)
        obs = pygame.surfarray.array3d(self.observation_screens[i])[::-1]
        obs = np.rot90(obs, 3)

        # Grayscale converting
        obs = obs[..., 0] * 0.299 + obs[..., 1] * 0.587 + obs[..., 2] * 0.114
        # obs = np.dot(obs[..., :3], [0.299, 0.587, 0.114])
        # np.testing.assert_almost_equal(obs, new_obs)

        # Cropping
        # obs = obs[15:70, 20:75]
        obs = np.reshape(obs, self.observation_space.shape)
        return obs

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        pygame.display.quit()
        pygame.quit()

    def render_indicators_for_pygame(self, screen, width=width, height=height, scale=30):
        if self.camera_follow < 0:
            return
        s = width / 40.0
        h = height / 40.0

        true_speed = np.sqrt(
            np.square(self.cars[self.camera_follow].hull.linearVelocity[0])
            + np.square(self.cars[self.camera_follow].hull.linearVelocity[1])
        )
        vertical_ind(screen, 0, height - 4 * h, width, 4 * h, 1000, (0, 0, 0))
        vertical_ind(screen, 5 * s, height - h, s, h, -0.02 * true_speed, (0, 0, 255))
        vertical_ind(screen, 7 * s, height - h, s, h, -0.01 * self.cars[self.camera_follow].wheels[0].omega,
                     (0.0, 0, 255))  # ABS sensors
        vertical_ind(screen, 8 * s, height - h, s, h, -0.01 * self.cars[self.camera_follow].wheels[1].omega,
                     (0.0, 0, 255))
        vertical_ind(screen, 9 * s, height - h, s, h, -0.01 * self.cars[self.camera_follow].wheels[2].omega,
                     (0.2 * 255, 0, 255))
        vertical_ind(screen, 10 * s, height - h, s, h, -0.01 * self.cars[self.camera_follow].wheels[3].omega,
                     (0.2 * 255, 0, 255))
        horiz_ind(screen, 20 * s, height - 2 * h, s, 2 * h, 10.0 * self.cars[self.camera_follow].wheels[0].joint.angle,
                  (0, 255, 0))
        horiz_ind(screen, 30 * s, height - 2 * h, s, 2 * h, 0.8 * self.cars[self.camera_follow].hull.angularVelocity,
                  (255, 0, 0))
        draw_text(screen, str("%05.0f" % self.rewards[self.camera_follow]), width / 100, height - height / 20, scale)

    def render_road_for_pygame(self, screen, width=width, height=height):
        screen.fill((0.4 * 255, 0.8 * 255, 0.4 * 255))
        tmp = Box2D.b2Transform()
        tmp.position = (0, 0)
        tmp.angle = -self.camera_angle
        k = PLAYFIELD / 20.0
        square_to_draw = []
        for x in range(-20, 20, 2):
            for y in range(-20, 20, 2):
                square_to_draw.append([(k * x + k, k * y + 0),
                                       (k * x + 0, k * y + 0),
                                       (k * x + 0, k * y + k),
                                       (k * x + k, k * y + k)])

        for square in square_to_draw:
            path = [self.camera_scale * (tmp * (-v[0] + self.camera_offset[0], -v[1] + self.camera_offset[1])) + (
                width / 2, height / 2) for v in square]
            to_print = False
            for pos in path:
                if pos[0] > -0 and pos[0] < width and pos[1] > -0 and pos[1] < height:
                    to_print = True
            if to_print:
                pygame.draw.polygon(screen, (0.4 * 255, 0.9 * 255, 0.4 * 255), path)

        for poly, color in self.road_poly:
            # path_tmp = [(-v[0] + self.camera_offset[0], -v[1] + self.camera_offset[1]) for v in poly]
            path = [self.camera_scale * (tmp * (-v[0] + self.camera_offset[0], -v[1] + self.camera_offset[1])) + (
                width / 2, height / 2) for v in poly]
            to_print = False
            for pos in path:
                if pos[0] > -0 and pos[0] < width and pos[1] > -0 and pos[1] < height:
                    to_print = True
            if to_print:
                pygame.draw.polygon(screen, [255 * i for i in color], path)

    def render_road_for_world_map(self):
        screen = pygame.Surface(self.world_size)
        screen.fill((0.4 * 255, 0.8 * 255, 0.4 * 255))
        k = PLAYFIELD / 20.0
        square_to_draw = []
        for x in range(-20, 20, 2):
            for y in range(-20, 20, 2):
                square_to_draw.append([(k * x + k, k * y + 0),
                                       (k * x + 0, k * y + 0),
                                       (k * x + 0, k * y + k),
                                       (k * x + k, k * y + k)])

        for square in square_to_draw:
            path = [
                (self.world_scale * -v[0] + self.world_size[0] / 2, self.world_scale * -v[1] + self.world_size[1] / 2)
                for v in square]
            pygame.draw.polygon(screen, (0.4 * 255, 0.9 * 255, 0.4 * 255), path)

        for poly, color in self.road_poly:
            path = [
                (self.world_scale * -v[0] + self.world_size[0] / 2, self.world_scale * -v[1] + self.world_size[1] / 2)
                for v in poly]
            pygame.draw.polygon(screen, [255 * i for i in color], path)
        return screen

    def render_road_for_observation_map(self):
        screen = pygame.Surface(self.world_size)
        screen.fill((0.4 * 255, 0.8 * 255, 0.4 * 255))
        k = PLAYFIELD / 20.0
        square_to_draw = []
        for x in range(-20, 20, 2):
            for y in range(-20, 20, 2):
                square_to_draw.append([(k * x + k, k * y + 0),
                                       (k * x + 0, k * y + 0),
                                       (k * x + 0, k * y + k),
                                       (k * x + k, k * y + k)])

        for square in square_to_draw:
            path = [
                (self.obs_scale * -v[0] + self.world_size[0] / 2, self.obs_scale * -v[1] + self.world_size[1] / 2)
                for v in square]
            pygame.draw.polygon(screen, (0.4 * 255, 0.9 * 255, 0.4 * 255), path)

        for poly, color in self.road_poly:
            path = [
                (self.obs_scale * -v[0] + self.world_size[0] / 2, self.obs_scale * -v[1] + self.world_size[1] / 2)
                for v in poly]
            pygame.draw.polygon(screen, [255 * i for i in color], path)
        return screen

    def render_cars_for_world_map(self, screen):
        # surface = pygame.Surface([self.world_size[0], self.world_size[1]], pygame.SRCALPHA, 32)
        for i in range(len(self.cars)):
            self.cars[i].draw_for_world_map(screen, self.camera_scale, self.world_size[0], self.world_size[1])

        return screen

    def camera_view(self, surface, screen, mode="human"):

        pos = self.camera_offset
        angle = self.camera_angle

        if mode == "human":
            width, height = self.window_size
            pos = (
                self.world_scale * -pos[0] + self.world_size[0] / 2,
                self.world_scale * -pos[1] + self.world_size[1] / 2)
        elif mode == "rgb_array":
            width = STATE_W
            height = STATE_H
            pos = (
                self.obs_scale * -pos[0] + self.world_size[0] / 2, self.obs_scale * -pos[1] + self.world_size[1] / 2)

        else:
            raise ValueError("Wrong mode {} in camera_view".format(mode))

        rect = pygame.Rect(pos[0] - width, pos[1] - height, 2 * width, 2 * height)

        camera_view = surface.subsurface(rect)
        camera_view.convert_alpha()
        camera_view = pygame.transform.rotate(camera_view, 57.295779513 * angle)
        center = camera_view.get_rect().center
        screen.blit(camera_view, (-center[0] + width / 2, -center[1] + height / 2))

    def camera_update(self, mode="human"):
        if (self.camera_follow != -1):
            angle = self.cars[self.camera_follow].hull.angle
            vel = self.cars[self.camera_follow].hull.linearVelocity
            if vel[0] ** 2 + vel[1] ** 2 > 0.5 ** 2:
                angle = math.atan2(-vel[0], +vel[1])

            tmp = Box2D.b2Transform()
            tmp.position = (0, 0)
            tmp.angle = angle
            if mode == "rgb_array":
                self.camera_offset = self.cars[self.camera_follow].hull.position + tmp * (0, 16)
                self.camera_angle = angle
                self.camera_scale = self.obs_scale
            elif mode == "human":
                self.camera_offset = self.cars[self.camera_follow].hull.position + tmp * (0, height / 54)
                self.camera_angle = angle
                self.camera_scale = self.world_scale
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
            if input_number == -3:
                self.isopen = False

    def render(self, mode="human", playground=None, playground_surface=None, drawing_for_player_num=None,
               agent_index=0):
        assert self.camera_offset is not None, "You should reset the environment before rendering!"
        if mode in ["human", "rgb_array"]:
            self.camera_update()
            playground = self.playground
            playground_surface = self.playground_surface
            self.camera_view(playground, playground_surface)
            for car in self.cars:
                car.draw_for_pygame(playground_surface, width, height, offset=self.camera_offset,
                                    angle=self.camera_angle,
                                    scale=self.camera_scale, mode="human")
            self.render_indicators_for_pygame(playground_surface, width=width, height=height)
            self.screen.blit(playground_surface, (0, 0))

            if mode == "rgb_array":
                obs = pygame.surfarray.array3d(self.screen)[::-1]
                obs = np.rot90(obs, 3)
                return obs
            else:
                pygame.display.flip()

        elif mode == "internal_rgb_array":
            self.camera_view(playground, playground_surface, mode="rgb_array")
            for i in range(self.num_player):
                self.cars[i].draw_for_pygame(playground_surface, STATE_W, STATE_H, offset=self.camera_offset,
                                             angle=self.camera_angle,
                                             scale=self.camera_scale, main_car_color=(drawing_for_player_num == i))
            self.render_indicators_for_pygame(playground_surface, width=STATE_W, height=STATE_H, scale=5)

        else:
            raise ValueError("Unknown rendering mode: {}".format(mode))

    def show_all_obs(self, obs, grayscale=False):
        for i in range(self.num_player):
            if grayscale:
                grayobs = np.dot(obs[i][..., :3], [0.299, 0.587, 0.114])
                plt.imshow(grayobs, cmap=plt.get_cmap('gray'))
            else:
                plt.imshow(obs[i])
            plt.show()
        self.show_all_car_obs = False

# if __name__ == "__main__":
#     env = CarRacing(num_player=num_player)
#     # example: env.reset(use_local_track="./track/test.json",record_track_to="")
#     # example: env.reset(use_local_track="",record_track_to="./track")
#     env.reset(use_local_track="",record_track_to="")
#     a = [[0.0, 0.0, 0.0] for _ in range(num_player)]
#     clock = pygame.time.Clock()
#     while True:
#         env.manage_input(key_phrase(a))
#         env.render()
#         observation, reward, done, info = env.step(a)
#         if env.show_all_car_obs:
#             env.show_all_obs(observation)
#         clock.tick(60)
#         fps = clock.get_fps()
#         print(fps)
