from object.dofus_agent import Agent
from object.dofus_status_window import DofusStatusWindow

agent = Agent(learning=False)
agent.load("./results/dofus - 2024-02-10 00:16.qtable")

window = DofusStatusWindow(agent, sleep_time=0.1)
window.run()
