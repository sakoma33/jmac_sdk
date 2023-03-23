import json
import os

import numpy as np


def evaluate_from_nall_logs():
    # ディレクトリのパスを指定
    dir_path = "logs"

    # ディレクトリ直下のディレクトリを取得
    dir_list = next(os.walk(dir_path))[1]

    # print(dir_list)
    dir_list = os.listdir(dir_path)

    # print(dir_list)

    dir_paths = [os.path.join(dir_path, dir_name, "player_0.json")
                 for dir_name in dir_list]
    player_0s = []
    final_tens = []
    # print(dir_paths)

    for dir_path in dir_paths:
        with open(dir_path, mode="r") as f:
            d = json.load(f)
            who = d.get("who", 0)
            final_tens.append(d["roundTerminal"]["finalScore"]["tens"][who])
            # print(who)
    print(final_tens)
    # print(len(final_tens))
    print(np.array(final_tens).mean())


def evaluate_from_arg(dir_list):
    # ディレクトリのパスを指定
    # dir_path = "logs"

    # # ディレクトリ直下のディレクトリを取得
    # dir_list = next(os.walk(dir_path))[1]

    # # print(dir_list)
    # dir_list = os.listdir(dir_path)

    # print(dir_list)

    dir_paths = [os.path.join(dir_path, "player_0.json")
                 for dir_path in dir_list]
    player_0s = []
    final_tens = []
    # print(dir_paths)

    for dir_path in dir_paths:
        with open(dir_path, mode="r") as f:
            d = json.load(f)
            who = d.get("who", 0)
            final_tens.append(d["roundTerminal"]["finalScore"]["tens"][who])
            # print(who)
    print(final_tens)
    # print(len(final_tens))
    print(np.array(final_tens).mean())
