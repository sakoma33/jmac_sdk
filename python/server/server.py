import eventlet
import socketio
from datetime import datetime
import json
import random

import mjx

json_open = open('samples/test.json', 'r')
content = json.load(json_open)

# 時間付きでの出力
def print_log(message):
    print('[{}] {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), message))

# ルームごとに接続されているクライアントのリスト
clients = {}

# ルームごとの卓
envs = {}

player_names_to_idx ={
    "player_0": 0,
    "player_1": 1,
    "player_2": 2,
    "player_3": 3,
}

'''
ルームごとにjsonを送る: self.emit('response', msg, room=roomId)

特定のクライアントにjsonを送る: self.emit('response', msg, room=clients[roomId][0])
(roomにクライアントのsidを指定する.)
'''

def play(roomId, server):
    """
    Args:
        roomId: int
        server: SocketIOServer
    """
    # プレイヤーの位置を決める (ランダム)
    players = random.sample(cliets[roomId], len(cliets[roomId]))

    # 卓の初期化
    env_ = envs[roomId]
    obs_dict = env_.reset()

    while not env.done():
        actions = {}
        for player_id, obs in obs_dict.items():
            sid = players[player_names_to_idx[player_id]]
            server.sio.emit('receive_content', obs_dict[player_id].to_proto(), room=sid, namespace='/test')
            actions[player_id] = None  # TODO: ここに `sid` のクライアンドからの行動結果が入る`
            obs_dict = env.step(actions)

    # 対局終了時にログを保存

# 名前空間を設定するクラス
class MyCustomNamespace(socketio.Namespace):

    # クライアントが接続したときに実行される関数
    def on_connect(self, sid, environ):
        print_log('connet sid : {}'.format(sid))
        print_log('connet env : {}'.format(environ))

    # クライアントを入室させる
    def on_enter_room(self, sid, roomId):
        self.enter_room(sid, roomId)
        if roomId not in clients:
            clients[roomId] = []
        clients[roomId].append(sid)
        # 4人入ったら対局開始
        if (len(clients[roomId]) == 4):
            env[roomId] = mjx.MjxEnv()

        print_log('enter room: {}'.format(roomId))

    # 送信してきたクライアントだけにメッセージを送る関数
    def on_sid_message(self, sid, msg):
        self.emit('response', msg, room=sid)
        print_log('emit sid : {}'.format(msg))

    # 送信してきたクライアントを除く同じルームのクライアントにメッセージを送信する関数
    def on_skip_sid_message(self, sid, data):
        self.emit('response', data["content"], room=data["roomId"], skip_sid=sid)
        print_log('emit skip sid : {}'.format(data["content"]))

    # 同じルームのすべてのクライアントにメッセージを送る関数
    def on_broadcast_message(self, sid, data):
        self.emit('response', data["content"], room=data["roomId"])
        print_log('emit all in {}: {}'.format(data["roomId"], data["content"]))

    # jsonが送信されたときに、送信してきたクライアントを除く同じルームのクライアントにjsonを送信する関数
    def on_send_json(self, sid, data):
        self.emit('receive_content', data["content"], room=data["roomId"], skip_sid=sid)
        print_log('emit skip sid : {}'.format(content))

    # 送信してきたクライアントだけにjsonを送る関数
    def on_receive_json(self, sid):
        self.emit('receive_content', content, room=sid)
        print_log('emit sid : {}'.format(content))

    # クライアントとの接続が切れたときに実行される関数
    def on_disconnect(self, sid):
        print_log('disconnect')

class SocketIOServer:

    def __init__(self, host, port, path):
        self.host = host
        self.port = port
        self.path = path
        self.sio = socketio.Server(cors_allowed_origins='*') # CORSのエラーを無視する設定


    def start(self, roomId):
        self.sio.register_namespace(MyCustomNamespace(self.path)) # 名前空間を設定
        app = socketio.WSGIApp(self.sio) # wsgiサーバーミドルウェア生成
        self.sio.start_background_task(self.actively_send_json, roomId) # バックグラウンドタスクの登録
        eventlet.wsgi.server(eventlet.listen((self.host, self.port)), app) # wsgiサーバー起動

    # 能動的にjsonを送信する
    def actively_send_json(self, roomId):
        while(True):
            self.sio.sleep(3)
            print_log('emit json actively in {}'.format(roomId))
            self.sio.emit('receive_content', content, room="123", namespace='/test')

if __name__ == '__main__':
    sio_server = SocketIOServer('localhost', 5000, '/test') # SocketIOClientクラスをインスタンス化
    sio_server.start("123") # サーバーを起動する（引数はjsonを送信するルームID）
