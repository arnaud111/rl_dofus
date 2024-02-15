from datetime import datetime

from matplotlib import pyplot as plt
from numpy import mean

from object.dofus_agent import Agent
from utils_dofus import REWARD_END_FIGHT

agent = Agent()
# agent.load("./results/dofus - 2024-02-09 21:01.qtable")
history = []
agent.score = -1
log_step_range = 1000
last_registered_time = datetime.now()

for i in range(int(1e6)):
    map_state = agent.reset()

    while True:
        map_state, reward = agent.do(map_state)
        if reward == REWARD_END_FIGHT:
            history.append(agent.score)
            break
    if i % log_step_range == 0 and i != 0:
        print(
            f"Iterations : {i}, "
            f"Best score : {max(history[i - log_step_range:i])}, "
            f"Lowest score : {min(history[i - log_step_range:i])}, "
            f"Mean : {mean(history[i - log_step_range:i])}, "
            f"Time elapsed : {datetime.now() - last_registered_time}")
        last_registered_time = datetime.now()

agent.save()
plt.plot(history)
plt.show()
