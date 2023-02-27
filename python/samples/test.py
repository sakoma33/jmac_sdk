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

class ShantenAgent(mjx.Agent):
    """A rule-based agent, which plays just to reduce the shanten-number.
    The logic is basically intended to reproduce Mjai's ShantenPlayer.
    - Mjai https://github.com/gimite/mjai
    """

    def __init__(self) -> None:
        super().__init__()

    def act(self, observation: mjx.Observation) -> mjx.Action:
        legal_actions = observation.legal_actions()
        if len(legal_actions) == 1:
            return legal_actions[0]

        # if it can win, just win
        win_actions = [a for a in legal_actions if a.type() in [mjx.ActionType.TSUMO, mjx.ActionType.RON]]
        if len(win_actions) >= 1:
            assert len(win_actions) == 1
            return win_actions[0]

        # if it can declare riichi, just declar riichi
        riichi_actions = [a for a in legal_actions if a.type() == mjx.ActionType.RIICHI]
        if len(riichi_actions) >= 1:
            assert len(riichi_actions) == 1
            return riichi_actions[0]

        # if it can apply chi/pon/open-kan, choose randomly
        steal_actions = [
            a
            for a in legal_actions
            if a.type() in [mjx.ActionType.CHI, mjx.ActionType.PON, mjx.ActionType, mjx.ActionType.OPEN_KAN]
        ]
        if len(steal_actions) >= 1:
            return random.choice(steal_actions)

        # if it can apply closed-kan/added-kan, choose randomly
        kan_actions = [
            a for a in legal_actions if a.type() in [mjx.ActionType.CLOSED_KAN, mjx.ActionType.ADDED_KAN]
        ]
        if len(kan_actions) >= 1:
            return random.choice(kan_actions)

        # discard an effective tile randomly
        legal_discards = [
            a for a in legal_actions if a.type() in [mjx.ActionType.DISCARD, mjx.ActionType.TSUMOGIRI]
        ]
        effective_discard_types = observation.curr_hand().effective_discard_types()
        effective_discards = [
            a for a in legal_discards if a.tile().type() in effective_discard_types
        ]
        if len(effective_discards) > 0:
            return random.choice(effective_discards)

        # if no effective tile exists, discard randomly
        return random.choice(legal_discards)


agent = ShantenAgent()
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
print(urls[0])
