import psycopg2
from utils_dofus import *


class DatabaseConnection:

    def __init__(self):
        self.dbname = 'rl_dofus'
        self.user = 'nono'
        self.password = 'azeAZE123'
        self.host = 'localhost'

        self.connection = psycopg2.connect(dbname=self.dbname, user=self.user, password=self.password, host=self.host)

        self.cursor = self.connection.cursor()

    def insert_new_state(self, state):
        for action in range(len(ACTIONS)):
            self.cursor.execute("INSERT INTO qtable(states, actions, reward) VALUES (%s, %s, 0)", (state, action))
        self.connection.commit()

    def get_max_action_state(self, state):
        self.cursor.execute("SELECT actions FROM qtable WHERE states = %s ORDER BY reward DESC", (state,))
        returned = self.cursor.fetchall()
        if len(returned) == 0:
            return None
        return returned[0][0]

    def add_reward(self, state, action, reward):
        self.cursor.execute("UPDATE qtable SET reward = reward + %s WHERE states = %s AND actions = %s RETURNING reward",
                       (reward, state, action))

        returned = self.cursor.fetchall()
        self.connection.commit()

        return returned[0][0]

    def compute_reward(self, state, action, reward, last):
        self.cursor.execute(
            "UPDATE qtable SET reward = reward + %s *(%s + %s * %s - reward) WHERE states = %s AND actions = %s RETURNING reward",
            (learning_rate, reward, decrease_factor, last, state, action))

        returned = self.cursor.fetchall()
        self.connection.commit()

        return returned[0][0]
