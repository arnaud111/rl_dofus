import random
import math
from collections import Counter
from PIL import Image

EMPTY_CELL = 0
WALL = 1
USABLE_CELL = 2
PLAYER = 3
MONSTER = 4
INVO = 5
BOSS = 6
UNDEFINED = -1

list_cell_color = {
    (0, 0, 0, 255): EMPTY_CELL,
    (142, 134, 94, 255): USABLE_CELL,
    (150, 142, 103, 255): USABLE_CELL,
    (85, 121, 56, 255): USABLE_CELL,
    (90, 125, 62, 255): USABLE_CELL,
    (88, 83, 58, 255): WALL,
    (218, 57, 45, 255): INVO,
    (46, 54, 61, 255): MONSTER,
    (251, 241, 191, 255): BOSS,
    (196, 19, 0, 255): PLAYER,
}

MAX_PA = 6
MAX_PM = 3

REWARD_MOB = 16
REWARD_END_TURN = -2
REWARD_MOVE = -4
REWARD_IMPOSSIBLE = -128
REWARD_END_FIGHT = 64

learning_rate = 1
decrease_factor = 0.5
random_action = 0.05

l = 20
h = 14

base_x = 235
base_y = 5

step_x = 130
step_y = 65

box_size = 30

dict_swap_dofus_to_pos = {}
dict_swap_pos_to_dofus = {}

dict_swap_dofus_to_array_index = {}
dict_swap_pos_to_array_index = {}

dict_swap_array_index_to_dofus = []
dict_swap_array_index_to_pos = []
index = 0

for j in range(h):

    x_start = 13 - j

    for i in range(l):
        dict_swap_dofus_to_pos[(j, i, False)] = (x_start + i, i + j)
        dict_swap_dofus_to_pos[(j, i, True)] = (x_start + i, i + j + 1)

        dict_swap_pos_to_dofus[(x_start + i, i + j)] = (j, i, False)
        dict_swap_pos_to_dofus[(x_start + i, i + j + 1)] = (j, i, True)

        dict_swap_dofus_to_array_index[(j, i, False)] = index
        dict_swap_pos_to_array_index[(x_start + i, i + j)] = index

        dict_swap_array_index_to_dofus.append((j, i, False))
        dict_swap_array_index_to_pos.append((x_start + i, i + j))
        index += 1

        dict_swap_dofus_to_array_index[(j, i, True)] = index
        dict_swap_pos_to_array_index[(x_start + i, i + j + 1)] = index
        dict_swap_array_index_to_dofus.append((j, i, True))
        dict_swap_array_index_to_pos.append((x_start + i, i + j + 1))

        index += 1


def get_pos_at_range(r):
    list_pos = []

    for i in range(r + 1):
        if i != 0:
            list_pos.append((0, i))
            list_pos.append((0, -i))

        for j in range(r - i):
            list_pos.append((1 + j, i))
            list_pos.append((-1 - j, i))

            if i != 0:
                list_pos.append((1 + j, -i))
                list_pos.append((-1 - j, -i))

    return list_pos


END_TURN = -1
ACTIONS = get_pos_at_range(6)
ACTIONS.append(END_TURN)


def get_pos_at_range_possible(map_array, start, r):
    list_pos = get_pos_at_range(r)
    full_list_pos = []
    for i in range(len(list_pos)):

        list_pos[i] = list_pos[i][0] + start[0], list_pos[i][1] + start[1]

        if list_pos[i] not in dict_swap_pos_to_array_index.keys():
            continue

        if map_array[dict_swap_pos_to_array_index[list_pos[i]]] == USABLE_CELL:
            full_list_pos.append(list_pos[i])

    return full_list_pos


def get_map(path):
    dict_map = {}
    array_map = []
    list_state_map = []

    with open(path, 'r') as f:
        lines = f.readlines()

        for line in lines:
            array_map.append(list(map(int, line.split(','))))

    for pos in dict_swap_pos_to_dofus.keys():
        dict_map[pos] = array_map[pos[0]][pos[1]]

    for i in range(h):
        for j in range(l):
            list_state_map.append(dict_map[dict_swap_dofus_to_pos[(i, j, False)]])
            list_state_map.append(dict_map[dict_swap_dofus_to_pos[(i, j, True)]])

    return list_state_map


def get_random_start_map(map_array):
    list_keys_player = []
    list_keys_mob = []

    for key in range(len(map_array)):
        if map_array[key] == 3:
            list_keys_player.append(key)
        elif map_array[key] == 4:
            list_keys_mob.append(key)

    i = random.randint(0, len(list_keys_player) - 1)
    list_keys_player.remove(list_keys_player[i])

    for _ in range(4):
        i = random.randint(0, len(list_keys_mob) - 1)
        list_keys_mob.remove(list_keys_mob[i])

    for pos in list_keys_player:
        map_array[pos] = 2

    for pos in list_keys_mob:
        map_array[pos] = 2

    return map_array


def get_player_pos(map_array):
    for pos in range(len(map_array)):
        if map_array[pos] == 3:
            return dict_swap_array_index_to_pos[pos]

    return -1


def mob_turn(map_array, invo=False):
    list_mob = []

    for pos in range(len(map_array)):
        if map_array[pos] == MONSTER:
            list_mob.append(pos)

    for mob in list_mob:

        possible_pos = get_pos_at_range_possible(map_array, dict_swap_array_index_to_pos[mob], 3)

        if len(possible_pos) == 0:
            continue

        map_array[mob] = USABLE_CELL
        new_pos = dict_swap_pos_to_array_index[possible_pos[random.randint(0, len(possible_pos) - 1)]]
        map_array[new_pos] = MONSTER

    return map_array


def map_contains_mob(map_array):
    for pos in range(len(map_array)):
        if map_array[pos] == MONSTER or map_array[pos] == INVO or map_array[pos] == BOSS:
            return True

    return False


def ldv(map_array, from_pos, to_pos):
    diff = [to_pos[0] - from_pos[0], to_pos[1] - from_pos[1]]
    tmp_from = [from_pos[0], from_pos[1]]

    if abs(diff[0]) > abs(diff[1]):
        index_factor = 0
        index_float = 1
    else:
        index_factor = 1
        index_float = 0

    factor = diff[0] / abs(diff[index_factor]), diff[1] / abs(diff[index_factor])

    while tmp_from[0] != to_pos[0] or tmp_from[1] != to_pos[1]:

        if not float(tmp_from[index_float]).is_integer():

            up = [0, 0]
            down = [0, 0]

            up[index_factor] = tmp_from[index_factor]
            down[index_factor] = tmp_from[index_factor]

            up[index_float] = math.ceil(tmp_from[index_float])
            down[index_float] = math.floor(tmp_from[index_float])

            if tuple(up) in dict_swap_pos_to_array_index.keys():
                up_index = dict_swap_pos_to_array_index[tuple(up)]
                if map_array[up_index] == WALL or map_array[up_index] == MONSTER or map_array[up_index] == BOSS or \
                        map_array[up_index] == INVO:
                    return False

            if tuple(down) in dict_swap_pos_to_array_index.keys():
                down_index = dict_swap_pos_to_array_index[tuple(down)]
                if map_array[down_index] == WALL or map_array[down_index] == MONSTER or map_array[down_index] == BOSS or \
                        map_array[down_index] == INVO:
                    return False
        else:

            tmp_index = dict_swap_pos_to_array_index[tuple(tmp_from)]

            if map_array[tmp_index] == WALL or map_array[tmp_index] == MONSTER or map_array[tmp_index] == BOSS or \
                    map_array[tmp_index] == INVO:
                return False

        tmp_from[0] += factor[0]
        tmp_from[1] += factor[1]

    return True


def can_move(map_array, from_pos, to_pos):
    diff = [to_pos[0] - from_pos[0], to_pos[1] - from_pos[1]]
    tmp_from = [from_pos[0], from_pos[1]]

    if abs(diff[0]) > abs(diff[1]):
        index_factor = 0
        index_float = 1
    else:
        index_factor = 1
        index_float = 0

    factor = diff[0] / abs(diff[index_factor]), diff[1] / abs(diff[index_factor])

    while tmp_from[0] != to_pos[0] and tmp_from[1] != to_pos[1]:
        if not float(tmp_from[index_float]).is_integer():

            up = [0, 0]
            down = [0, 0]

            up[index_factor] = tmp_from[index_factor]
            down[index_factor] = tmp_from[index_factor]

            up[index_float] = math.ceil(tmp_from[index_float])
            down[index_float] = math.floor(tmp_from[index_float])

            if tuple(up) in dict_swap_pos_to_array_index.keys():
                up_index = dict_swap_pos_to_array_index[tuple(up)]
                if map_array[up_index] == WALL or map_array[up_index] == MONSTER or map_array[up_index] == BOSS or \
                        map_array[up_index] == INVO:
                    return False

            if tuple(down) in dict_swap_pos_to_array_index.keys():
                down_index = dict_swap_pos_to_array_index[tuple(down)]
                if map_array[down_index] == WALL or map_array[down_index] == MONSTER or map_array[down_index] == BOSS or \
                        map_array[down_index] == INVO:
                    return False

        else:

            tmp_index = dict_swap_pos_to_array_index[tuple(tmp_from)]

            if map_array[tmp_index] != USABLE_CELL:
                return False

        tmp_from[0] += factor[0]
        tmp_from[1] += factor[1]

    return True


def count_pm_to_pos(map_array, player_pos, expected_pos, max_pm):
    diff = expected_pos[0] - player_pos[0], expected_pos[1] - player_pos[1]
    dist = abs(diff[0]) + abs(diff[1])

    if dist < max_pm and can_move(map_array, player_pos, expected_pos):
        return dist

    return -1


def get_colors(path=None, img=None):
    if img is None:
        img = Image.open(path)

    pixels = list(img.getdata())

    colors_counter = Counter(pixels)
    colors_counter = sorted(colors_counter.items(), key=lambda x: x[1], reverse=True)

    color1, _ = colors_counter[0]
    if len(colors_counter) > 1:
        color2, _ = colors_counter[1]
    else:
        color2 = (0, 0, 0)

    return color1, color2


def get_image_type(path=None, img=None):
    if img is None:
        most = get_colors(path=path)
    else:
        most = get_colors(img=img)

    type = UNDEFINED

    if most[0] in list_cell_color.keys():
        type = list_cell_color[most[0]]
    else:
        print(most[0])

    if type == USABLE_CELL and most[1] == (46, 54, 61):
        type = MONSTER

    return type
