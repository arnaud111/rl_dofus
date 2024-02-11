import time

import arcade
import pyautogui

from object.dofus_agent import Agent
from object.map_state import MapState
from utils_dofus import base_x, base_y, step_y, step_x, box_size, ACTIONS, get_player_pos, END_TURN, \
    dict_swap_pos_to_array_index, USABLE_CELL, MONSTER, INVO, BOSS, count_pm_to_pos, PLAYER, dict_swap_pos_to_dofus

agent = Agent(learning=False)
agent.load("./results/dofus - 2024-02-10 00:16.qtable")


def can_play(screen):
    try:
        if pyautogui.locate("resources/end_turn.png", screen) is not None:
            return True
    except pyautogui.ImageNotFoundException:
        return False
    return False


def get_pa(screen):
    try:
        if pyautogui.locate("resources/pa_6.png", screen) is not None:
            return 6
        if pyautogui.locate("resources/pa_3.png", screen) is not None:
            return 3
        if pyautogui.locate("resources/pa_0.png", screen) is not None:
            return 0
    except pyautogui.ImageNotFoundException:
        return 0
    return 0


def get_pm(screen):
    try:
        if pyautogui.locate("resources/pm_3.png", screen) is not None:
            return 3
        if pyautogui.locate("resources/pm_2.png", screen) is not None:
            return 2
        if pyautogui.locate("resources/pm_1.png", screen) is not None:
            return 1
        if pyautogui.locate("resources/pm_0.png", screen) is not None:
            return 0
    except pyautogui.ImageNotFoundException:
        return 0
    return 0


def center_cell(dofus_pos):
    x = base_x + step_x * dofus_pos[0]
    y = base_y + step_y * dofus_pos[1]

    if dofus_pos[2]:
        x += step_x // 2
        y += step_y // 2

    x += box_size // 2
    y += box_size

    return x, y


def do_for_real(state, action):
    clicked = ACTIONS[action]
    player_pos = get_player_pos(state.map)

    if player_pos == -1:
        return None

    if clicked == END_TURN:
        pyautogui.keyDown("f1")
        time.sleep(0.01)
        pyautogui.keyUp("f1")
        return None

    clicked_pos = player_pos[0] + clicked[0], player_pos[1] + clicked[1]

    if clicked_pos not in dict_swap_pos_to_array_index.keys():
        return None

    clicked_index = dict_swap_pos_to_array_index[clicked_pos]

    if state.map[clicked_index] == MONSTER or state.map[clicked_index] == INVO or state.map[clicked_index] == BOSS:
        state.map[clicked_index] = USABLE_CELL
        pyautogui.keyDown("2")
        time.sleep(0.01)
        pyautogui.keyUp("2")
        time.sleep(0.1)
    elif state.map[clicked_index] == USABLE_CELL:
        pm_used = count_pm_to_pos(state.map, player_pos, clicked_pos, state.pm)
        if pm_used == -1 or pm_used > state.pm:
            return None
        state.pm -= pm_used
        state.map[clicked_index] = PLAYER
        state.map[dict_swap_pos_to_array_index[player_pos]] = USABLE_CELL

    pos = center_cell(dict_swap_pos_to_dofus[clicked_pos])
    pyautogui.click(pos)
    pyautogui.moveTo(100, 100, duration=0.1)
    # move_mouse(pos[0], pos[1])
    # click()
    # move_mouse(100, 100)

    return state


while 1:
    time.sleep(3)
    screenshot = pyautogui.screenshot()
    if can_play(screenshot):
        time.sleep(1)
        screenshot = pyautogui.screenshot()
        map_state = MapState.screen_info(get_pa(screenshot), get_pm(screenshot), screenshot=screenshot)
        while map_state is not None:
            action, _ = agent.best_action(map_state)
            map_state = do_for_real(map_state, action)
            time.sleep(1)
    else:
        print("Searching for action to do")
