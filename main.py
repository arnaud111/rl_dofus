from matplotlib import pyplot as plt

from object.dofus_agent import Agent
from object.dofus_status_window import DofusStatusWindow

agent = Agent()
agent.load("./results/dofus - 2024-02-10 00:16.qtable")

window = DofusStatusWindow(agent)
window.run()

plt.plot(window.history)
plt.show()
