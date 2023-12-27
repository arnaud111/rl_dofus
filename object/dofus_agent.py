from dofus_environement import Environment
from map_state import MapState
from database_connection import DatabaseConnection
from utils_dofus import ACTIONS, random_action
import random


class Agent:

    def __init__(self, database_connection, learning=True):
        self.database_connection = database_connection
        self.env = Environment()
        self.map_state = MapState.generate()
        self.score = 0
        self.learning = learning
        self.list_actions = []

    def reset(self):
        self.map_state = MapState.generate()
        self.score = 0
        if self.learning and len(self.list_actions) > 0:
            self.back_propagation()
        self.list_actions = []

    def do_all(self, state):
        for action in range(len(ACTIONS)):
            reward, new_state = self.env.do(self.map_state, action)
            self.database_connection.add_reward(state, action, reward)

    def best_action(self, state):

        action = self.database_connection.get_max_action_state(state)
        if action is None:
            self.database_connection.insert_new_state(state)
            self.do_all(state)
            action = self.database_connection.get_max_action_state(state)

        if self.learning and random.uniform(0, 1) < random_action:
            action = random.randint(0, len(ACTIONS) - 1)

        return action

    def back_propagation(self):

        last = self.database_connection.add_reward(self.list_actions[-1][0], self.list_actions[-1][1], self.list_actions[-1][2])
        for i in range(len(self.list_actions) - 2, -1, -1):
            last = self.database_connection.compute_reward(self.list_actions[i][0], self.list_actions[i][1], self.list_actions[i][2], last)

    def do(self):
        action = self.best_action(self.map_state.state)

        reward, new_state = self.env.do(self.map_state, action)

        self.list_actions.append((self.map_state.state, action, reward))

        self.map_state = new_state
        self.score += reward

        return reward
