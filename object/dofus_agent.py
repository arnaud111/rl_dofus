from datetime import datetime

from object.dofus_environement import Environment
from object.map_state import MapState
from utils_dofus import ACTIONS, random_action
import random
import pickle
from os.path import exists


class Agent:

    def __init__(self, learning=True, learning_rate=1, discount_factor=0.9):
        self.qtable = {}
        self.env = Environment()
        self.score = 0
        self.iteration = 0
        self.learning = learning
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

    def reset(self):
        self.iteration = 0
        self.score = 0
        return MapState.generate()

    def do_all(self, map_state):
        self.qtable[map_state.state] = {}
        for action in range(len(ACTIONS)):
            reward, new_state = self.env.do(map_state, action)
            self.qtable[map_state.state][action] = reward

    def best_action(self, map_state):
        if map_state.state not in self.qtable:
            self.do_all(map_state)

        action = max(self.qtable[map_state.state], key=self.qtable[map_state.state].get)

        if self.learning and random.uniform(0, 1) < random_action:
            return random.randint(0, len(ACTIONS) - 1), True

        return action, False

    def do(self, map_state):
        action, randomed = self.best_action(map_state)

        reward, new_state = self.env.do(map_state, action)

        if randomed and map_state.state == new_state.state:
            return new_state, reward

        if new_state.state not in self.qtable:
            self.do_all(new_state)
        maxQ = max(self.qtable[new_state.state].values())
        delta = self.learning_rate * (reward + self.discount_factor * maxQ - self.qtable[map_state.state][action])
        self.qtable[map_state.state][action] += delta

        self.score += reward
        self.iteration += 1

        return new_state, reward

    def load(self, filename):
        if exists(filename):
            with open(filename, 'rb') as file:
                self.qtable = pickle.load(file)
            self.reset()

    def save(self):
        now = datetime.now()
        with open(f"./results/dofus - {now.strftime('%Y-%m-%d %H:%M')}.qtable", 'wb') as file:
            pickle.dump(self.qtable, file)
