from utils_dofus import get_random_start_map, get_map, MAX_PA, MAX_PM, dict_swap_pos_to_dofus, base_x, step_y, step_x, \
    base_y, box_size, get_image_type, dict_swap_array_index_to_dofus, INVO, BOSS, MONSTER, UNDEFINED, EMPTY_CELL
import pyautogui


class MapState:

    def __init__(self, pa, pm, map, map_index):

        self.map = map
        self.map_index = map_index
        self.pa = pa
        self.pm = pm

    def clone(self):

        new_map = []

        for i in range(len(self.map)):
            new_map.append(self.map[i])

        return MapState(self.pa, self.pm, new_map, self.map_index)

    @staticmethod
    def generate():

        # self.map_index = random.randint(0, 4)
        map_index = 0
        map = get_random_start_map(get_map(f"./data/map_array/map_{map_index + 1}.txt"))

        return MapState(MAX_PA, MAX_PM, map, map_index)

    @staticmethod
    def screen_info(pa, pm, map_index=0, screenshot=None):

        array_map = []
        if screenshot is not None:
            screenshot = pyautogui.screenshot()

        for i in range(len(dict_swap_array_index_to_dofus)):

            dofus_pos = dict_swap_array_index_to_dofus[i]
            x = base_x + step_x * dofus_pos[0]
            y = base_y + step_y * dofus_pos[1]

            if dofus_pos[2]:
                x += step_x // 2
                y += step_y // 2

            cropped = screenshot.crop((x, y, x + box_size, y + box_size))

            image_type = get_image_type(img=cropped)

            if image_type == INVO or image_type == BOSS:
                image_type = MONSTER
            if image_type == UNDEFINED:
                image_type = EMPTY_CELL
            array_map.append(image_type)

        return MapState(pa, pm, array_map, map_index)

    @property
    def state(self):

        state = ""
        for i in range(len(self.map)):
            state += str(self.map[i])

        state += str(self.pa)
        state += str(self.pm)

        return state