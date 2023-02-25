import mjx
import random
from typing import List
from tqdm import tqdm

class RandomAgent(mjx.Agent):
    def __init__(self) -> None:
        super().__init__()

    def act(self, observation: mjx.Observation) -> mjx.Action:
        legal_actions: List[mjx.Action] = observation.legal_actions()
        return random.choice(legal_actions)

agent = RandomAgent()
env = mjx.MjxEnv()
N = 100
for _ in tqdm(range(N)):
  obs_dict = env.reset()
  while not env.done():
      actions = {player_id: agent.act(obs)
              for player_id, obs in obs_dict.items()}
      obs_dict = env.step(actions)
  returns = env.rewards()

print(obs_dict['player_0'].to_json())

env.state().save_svg("test.svg")
