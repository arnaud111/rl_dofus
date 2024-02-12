from datetime import datetime

from matplotlib import pyplot as plt
from object.dofus_agent import Agent
from utils_dofus import REWARD_END_FIGHT

agent = Agent()
# agent.load("./results/dofus - 2024-02-09 21:01.qtable")
history = []
agent.score = -1
last_registered_time = datetime.now()
for i in range(int(1e6)):

    map_state = agent.reset()

    while True:
        map_state, reward = agent.do(map_state)
        if reward == REWARD_END_FIGHT:
            history.append(agent.score)
            break
    if i % 1000 == 0:
        print(f"Iterations : {i}, best score : {max(history)}, time elapsed : {datetime.now() - last_registered_time}")
        last_registered_time = datetime.now()

agent.save()
plt.plot(history)
plt.show()
