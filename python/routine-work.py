import mjx
import random
from typing import List
from tqdm import tqdm

import time
import random


class RandomAgent(mjx.Agent):
    def __init__(self) -> None:
        super().__init__()

    def act(self, observation: mjx.Observation) -> mjx.Action:
        legal_actions: List[mjx.Action] = observation.legal_actions()
        return random.choice(legal_actions)


# send_obs関数の検証のため2回に1回Trueを返す関数を実装
def my_function():
    return random.randint(1, 2) == 1


# これを一旦定義しておいて， この関数を呼ぶことで通信できると仮定
def send_obs(player_id, obs):
    # レスポンスを待機する
    timeout = 5  # 秒
    start_time = time.time()
    while time.time() < start_time + timeout:
        if my_function():
            # print('レスポンスが正常に返ってきました。')
            break
        time.sleep(1)

    # レスポンスが返ってこない場合
    else:
        # print('レスポンスが返ってきませんでした。')
        return random.choice(obs.legal_actions())

    # レスポンスが返ってきた場合
    return random.choice(obs.legal_actions())


agent = RandomAgent()  # プレイヤー
env = mjx.MjxEnv()  # 卓

obs_dict = env.reset()  # 初期の盤面の状態(+最初のツモ)
while not env.done():
    actions = {player_id: send_obs(player_id, obs)
               for player_id, obs in obs_dict.items()}
    #  obs_dict.items() では，行動することができるプレイヤーのみ値を取り出せる? 全プレイヤー取り出せる?→行動ができるプレイヤーのみ値を取り出せる
    obs_dict = env.step(actions)

returns = env.rewards()
