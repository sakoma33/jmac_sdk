import mjx
import random
from typing import List
from tqdm import tqdm
from convertLog import ConvertLog

class RandomAgent(mjx.Agent):
    def __init__(self) -> None:
        super().__init__()

    def act(self, observation: mjx.Observation) -> mjx.Action:
        legal_actions: List[mjx.Action] = observation.legal_actions()
        return random.choice(legal_actions)


agent = RandomAgent()
env = mjx.MjxEnv()
N = 100
urls=[]
for _ in tqdm(range(N)):
    logs = ConvertLog()
    obs_dict = env.reset()
    while not env.done():
        actions = {player_id: agent.act(obs)
                for player_id, obs in obs_dict.items()}
        obs_dict = env.step(actions)
        if len(obs_dict.keys())==4:
            logs.add_log(obs_dict)
    urls.append(logs.get_url())
    returns = env.rewards()

env.state().save_svg("test.svg")
print(urls[-1])
