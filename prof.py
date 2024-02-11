import arcade
from random import *
import pickle
from os.path import exists
import matplotlib.pyplot as plt

MAZE = """
#.################
#                #
####   #         #
#      #         #
#                #
#   ######### ####
#           #    #
#       #   #    #
#       #   #    #
#  ######   #    #
#       #   ######
#       #        #
################*#
"""

ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT = 'U', 'D', 'L', 'R'
ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]

ACTION_MOVE = {ACTION_UP: (-1, 0),
               ACTION_DOWN: (1, 0),
               ACTION_LEFT: (0, -1),
               ACTION_RIGHT: (0, 1)}

REWARD_WALL = -128
REWARD_DEFAULT = -1
REWARD_GOAL = 64

MAP_WALL = '#'
MAP_START = '.'
MAP_GOAL = '*'

SPRITE_SCALING = 0.25
SPRITE_SIZE = 32

AGENT_FILE = 'agent.qtable'


class Environment:
    def __init__(self, str_map):
        row, col = 0, 0
        self.__map = {}
        for line in str_map.strip().split('\n'):
            for char in line:
                self.__map[row, col] = char
                if char == MAP_START:
                    self.start = (row, col)
                elif char == MAP_GOAL:
                    self.goal = (row, col)
                col += 1
            row += 1
            col = 0
            self.cols = len(line)
            self.rows = row

    def do(self, state, action):
        move = ACTION_MOVE[action]
        new_state = (state[0] + move[0], state[1] + move[1])

        if new_state not in self.states \
                or self.__map[new_state] in [MAP_WALL, MAP_START]:
            reward = REWARD_WALL
        else:
            state = new_state
            if self.__map[new_state] == MAP_GOAL:
                reward = REWARD_GOAL
            else:
                reward = REWARD_DEFAULT

        return state, reward

    @property
    def states(self):
        return self.__map.keys()

    def map(self, state):
        return self.__map[state]


def arg_max(table):
    # keys = list(table.keys())
    # best = keys[0]
    # for i in range(1, len(keys)):
    #    if table[keys[i]] > table[best]:
    #        best = keys[i]
    # return best

    return max(table, key=table.get)


class Agent:
    def __init__(self, env, learning_rate=1, discount_factor=0.9):
        self.env = env
        self.reset()
        self.qtable = {}
        for state in env.states:
            self.qtable[state] = {}
            for action in ACTIONS:
                self.qtable[state][action] = 0.0
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.history = []

    def reset(self):
        self.__state = env.start
        self.score = 0
        self.iteration = 0

    def best_action(self):
        action = arg_max(self.qtable[self.__state])
        return action

    def do(self):
        action = self.best_action()
        new_state, reward = self.env.do(self.__state, action)

        maxQ = max(self.qtable[new_state].values())
        delta = self.learning_rate * \
                (reward + self.discount_factor * \
                 maxQ - self.qtable[self.__state][action])

        self.qtable[self.__state][action] += delta

        self.__state = new_state
        self.score += reward
        self.iteration += 1

        if self.__state == self.env.goal:
            self.history.append(self.score)

        return action, reward

    def load(self, filename):
        if exists(filename):
            with open(filename, 'rb') as file:
                self.qtable = pickle.load(file)
            self.reset()

    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.qtable, file)

    @property
    def state(self):
        return self.__state


class MazeWindow(arcade.Window):
    def __init__(self, agent):
        super().__init__(agent.env.cols * SPRITE_SIZE,
                         agent.env.rows * SPRITE_SIZE, "ESGI Maze")
        self.agent = agent

    def setup(self):
        self.walls = arcade.SpriteList()

        for state in self.agent.env.states:
            sprite = \
                arcade.Sprite(":resources:images/tiles/brickTextureWhite.png",
                              SPRITE_SCALING)
            if self.agent.env.map(state) == MAP_WALL:
                sprite.center_x, sprite.center_y = self.state_to_xy(state)
            self.walls.append(sprite)

        self.goal = arcade.Sprite(":resources:images/tiles/signExit.png",
                                  SPRITE_SCALING)
        self.goal.center_x, self.goal.center_y = self.state_to_xy(self.agent.env.goal)

        self.start = arcade.Sprite(":resources:images/tiles/doorClosed_mid.png",
                                   SPRITE_SCALING)
        self.start.center_x, self.start.center_y = self.state_to_xy(self.agent.env.start)

        self.player = arcade.Sprite(":resources:images/enemies/bee.png",
                                    SPRITE_SCALING)
        self.update_player()

    def state_to_xy(self, state):
        return (state[1] + 0.5) * SPRITE_SIZE, \
               (self.agent.env.rows - state[0] - 0.5) * SPRITE_SIZE

    def on_draw(self):
        arcade.start_render()
        self.goal.draw()
        self.start.draw()
        self.walls.draw()
        self.player.draw()
        arcade.draw_text(f'#{self.agent.iteration} Score : {self.agent.score}',
                         10, 10, arcade.color.AUBURN, 16)

    def on_update(self, delta_time):
        if self.agent.state != self.agent.env.goal:
            action, reward = self.agent.do()
        else:
            self.agent.reset()

        self.update_player()

    def on_key_press(self, key, modifiers):
        if key == arcade.key.R:
            self.agent.reset()
            self.update_player()

    def update_player(self):
        self.player.center_x, self.player.center_y = self.state_to_xy(self.agent.state)


if __name__ == "__main__":
    env = Environment(MAZE)
    agent = Agent(env)
    agent.load(AGENT_FILE)

    window = MazeWindow(agent)
    window.setup()
    window.run()

    agent.save(AGENT_FILE)
    plt.plot(agent.history)
    plt.show()
