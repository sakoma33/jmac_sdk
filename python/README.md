# 実装例

## 本番での動作
- sample_client.py
- sample_server.py

実行例: `$ python sample_server.py` `$ python sample_client.py`

プレイヤーとなる sample_client.py を ，麻雀の卓となる sample_server.py に接続して対局を行います．
運営での順位決定時には，このプログラムで動作する必要があります．

よくわからない場合，このプログラム中の `MyAgent.custom_act()` を書き換える形で開発すると良いです．

## 開発時にたくさん対局を回したいとき
- sample_trial.py

実行例: `$ python sample_trial.py -l -n 16`
- `-l` : 対局のログを`./logs/`に出力
- `-n NUMBER` : 指定した回数だけ対局(半荘戦)を実行
