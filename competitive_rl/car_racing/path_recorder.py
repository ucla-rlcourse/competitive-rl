import json

from Box2D import Box2D, b2Vec2

from competitive_rl.car_racing.pygame_rendering import draw_dot


def path_record(path, position=None, angle=None, fixtures=None):
    if position != None:
        path["positions"].append(position)
    if angle != None:
        path["angles"].append(angle)
    if fixtures != None:
        path["fixtures"].append(fixtures)

    return path

def path_record_to_file(file_path, path):
    file = open((file_path), 'w', encoding='utf-8')
    json.dump(path, file)

def path_drawer(screen, path, camera_offset,camera_angle, scale, W_W, W_H):
    tmp = Box2D.b2Transform()
    tmp.position = (0, 0)
    tmp.angle = -camera_angle

    path_draw = [ -scale* (tmp * (b2Vec2(pos)- camera_offset)) + (W_W/2, W_H/2) for pos in path["positions"]]

    for i in range(len(path_draw)):
        draw_dot(screen, (255,255,255), (int(path_draw[i][0]), int(path_draw[i][1])), 3)