from matplotlib import pyplot as plt
from object.dofus_agent import Agent
from utils_dofus import REWARD_END_FIGHT

agent = Agent()
#agent.load("./results/dofus - 2024-02-09 21:01.qtable")
history = []
agent.score = -1
cnt = 0

for i in range(1000000):

    map_state = agent.reset()

    while True:
        map_state, reward = agent.do(map_state)
        if reward == REWARD_END_FIGHT:
            history.append(agent.score)
            break

    cnt += 1

agent.save()
plt.plot(history)
plt.show()
