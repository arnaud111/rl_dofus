import time

import arcade
from object.dofus_agent import Agent
from utils_dofus import box_size, dict_swap_array_index_to_dofus, base_x, base_y, step_x, step_y, REWARD_END_FIGHT


class DofusStatusWindow(arcade.Window):

    def __init__(self, agent, sleep_time=0):
        super().__init__(2256 // 2, 1504 // 2, "DofusStatus")
        arcade.set_background_color(arcade.color.BLACK)
        self.agent = agent
        self.map_state = self.agent.reset()
        self.cnt = 0
        self.history = []
        self.sleep_time = sleep_time

    def draw_box(self, x, y, color):
        arcade.draw_rectangle_filled(x, 1504 // 2 - y, box_size // 2, box_size // 2, color)

    def draw_score(self, x, y, score):
        arcade.draw_text(f"Score : {score}", start_x=x, start_y=y, font_size=16, bold=True)

    def draw_legends(self):
        arcade.draw_rectangle_filled(100, 100, box_size // 2, box_size // 2, arcade.color.BLUE)
        arcade.draw_text("AGENT", start_x=120, start_y=93, font_size=12, bold=True)
        arcade.draw_rectangle_filled(100, 80, box_size // 2, box_size // 2, arcade.color.GRAY)
        arcade.draw_text("WALL", start_x=120, start_y=73, font_size=12, bold=True)
        arcade.draw_rectangle_filled(100, 60, box_size // 2, box_size // 2, arcade.color.RED)
        arcade.draw_text("ENEMY", start_x=120, start_y=53, font_size=12, bold=True)
        arcade.draw_rectangle_filled(100, 40, box_size // 2, box_size // 2, arcade.color.GREEN)
        arcade.draw_text("GRASS", start_x=120, start_y=33, font_size=12, bold=True)
        arcade.draw_text("VOID", start_x=120, start_y=13, font_size=12, bold=True)

    def on_draw(self):
        arcade.start_render()
        self.draw_score(90, 120, self.agent.score)
        self.draw_legends()
        for pos in range(len(self.map_state.map)):
            if self.map_state.map[pos] != 0:

                dofus_pos = dict_swap_array_index_to_dofus[pos]

                x = base_x + step_x * dofus_pos[0]
                y = base_y + step_y * dofus_pos[1]

                if dofus_pos[2]:
                    x += step_x // 2
                    y += step_y // 2

                x = x // 2
                y = y // 2

                if self.map_state.map[pos] == 1:
                    self.draw_box(x, y, arcade.color.GRAY)
                elif self.map_state.map[pos] == 2:
                    self.draw_box(x, y, arcade.color.GREEN)
                elif self.map_state.map[pos] == 3:
                    self.draw_box(x, y, arcade.color.BLUE)
                elif self.map_state.map[pos] == 4:
                    self.draw_box(x, y, arcade.color.RED)
                elif self.map_state.map[pos] == 5:
                    self.draw_box(x, y, arcade.color.YELLOW)
                elif self.map_state.map[pos] == 6:
                    self.draw_box(x, y, arcade.color.ORANGE)
                else:
                    self.draw_box(x, y, arcade.color.AQUA)

    def on_key_press(self, key, modifiers):
        if key == arcade.key.S:
            self.agent.save()

    def on_update(self, delta_time):
        time.sleep(self.sleep_time)
        self.map_state, r = self.agent.do(self.map_state)
        if r == REWARD_END_FIGHT:
            self.cnt += 1
            self.history.append(self.agent.score)
            self.map_state = self.agent.reset()
