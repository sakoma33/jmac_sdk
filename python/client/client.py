import socketio
import time
from datetime import datetime
import json
import argparse

json_open = open('test.json', 'r')
content = json.load(json_open)

# 時間付きでの出力
def print_log(message):
    print('[{}] {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), message))

# 名前空間を設定するクラス
class MyCustomNamespace(socketio.ClientNamespace):
    def __init__(self, namespace=None, agent=None):
        super().__init__(namespace)

        self.agent = agent  # TODO: ここスッキリさせる

    def on_connect(self):
        print_log('connect')

    def on_disconnect(self):
        print_log('disconnect')

    # def on_enter_room(self, room):
    #     self.emit('send_roomid', room)
    #     print_log('enter room:'+room)

    def on_response(self, msg):
        print_log('response : {}'.format(msg))

    # jsonが送られてきたときの処理
    def on_receive_content(self, content):
        print_log('json response : {}'.format(content))

    def on_ask_act(self, content):
        print(content)
        print(type(content))
        obs = content  # TODO: 多分形式の変換が必要 json -> ??? or proto ->
        decided_action = self.agent.act(obs)
        return decided_action




class SocketIOClient:

    def __init__(self, host, path, roomId):
        self.host = host
        self.path = path
        self.sio = socketio.Client()
        self.roomId = roomId

    def connect(self):
        self.sio.register_namespace(MyCustomNamespace(self.path)) # 名前空間を設定
        self.sio.connect(self.host) # サーバーに接続
        self.sio.emit('enter_room', self.roomId, namespace = self.path) # ルームIDを指定して入室
        self.sio.start_background_task(self.my_background_task, 123) # バックグラウンドタスクの登録 (123は引数の書き方の参考のため、処理には使っていない)
        self.sio.wait() # イベントが待ち

    def my_background_task(self, my_argument): # ここにバックグランド処理のコードを書く
        while True:
            input_data = input("send data:") # ターミナルから入力された文字を取得

            if input_data == 'send json':
                # jsonを送信する
                data = {
                    "roomId": self.roomId,
                    "content": content
                }
                self.sio.emit('send_json', data, namespace = self.path)
            elif input_data == 'receive json':
                # jsonを受信する
                self.sio.emit('receive_json', namespace = self.path)
            else:
                # メッセージを送信する
                data = {
                    "roomId": self.roomId,
                    "content": input_data
                }
                self.sio.emit('broadcast_message', data, namespace = self.path) # ターミナルで入力された文字をサーバーに送信
            self.sio.sleep(1)

if __name__ == '__main__':
    # roomIDを取得
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--room', type=str)
    args = parser.parse_args()
    print_log(args.room)

    sio_client = SocketIOClient('http://localhost:5000', '/test', args.room) # SocketIOClientクラスをインスタンス化
    sio_client.connect()
