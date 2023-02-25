from abc import abstractmethod

import mjx


class CustomAgentBase(mjx.Agent):
    def __init__(self):
        super.__init__()

    @abstractmethod
    def custom_act(self, obs: mjx.Observation) -> mjx.Action:
        """参加者はこの関数をオーバーライドして行動を実装する"""
        pass

    def act(self, observation: mjx.Observation) -> mjx.Action:
        # 参加者が実装した関数でエラーが起きた場合は
        # - ツモ時: ツモ切り
        # - ポンとかの選択時: パス
        # をする
        try:
            return self.custom_act(observation)
        except:
            legal_actions = observation.legal_actions()
            if len(legal_actions) == 1:
                return legal_actions[0]
            for action in legal_actions:
                if action.type() in [mjx.ActionType.TSUMOGIRI, mjx.ActionType.PASS]:
                    return action
