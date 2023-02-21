import eventlet
import socketio
from datetime import datetime
import json

json_open = open('test.json', 'r')
content = json.load(json_open)


# 名前空間を設定するクラス
class MyCustomNamespace(socketio.Namespace): 

    # クライアントが接続したときに実行される関数
    def on_connect(self, sid, environ):
        print('[{}] connet sid : {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S') , sid))
        print('[{}] connet env : {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S') , environ))
            
    # 送信してきたクライアントだけにメッセージを送る関数
    def on_sid_message(self, sid, msg): 
        self.emit('response', msg, room=sid)
        print('[{}] emit sid : {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S') , msg))

    # 送信してきたクライアントを除く全ての接続しているクライアントにメッセージを送信する関数
    def on_skip_sid_message(self, sid, msg):
        self.emit('response', msg, skip_sid=sid) 
        print('[{}] emit skip sid : {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S') , msg))

    # 接続しているすべてのクライアントにメッセージを送る関数
    def on_broadcast_message(self, sid, msg):
        self.emit('response', msg)
        print('[{}] emit all : {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S') , msg))

    # jsonが送信されたときに、送信してきたクライアントを除く全てのクライアントにjsonを送信する関数
    def on_send_json(self, sid, content):
        self.emit('receive_content', content, skip_sid=sid)
        print('[{}] emit skip sid : {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S') , content))

    # 送信してきたクライアントだけにjsonを送る関数
    def on_receive_json(self, sid, content):
        self.emit('receive_content', content, room=sid)
        print('[{}] emit sid : {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S') , content))
    
    # クライアントとの接続が切れたときに実行される関数
    def on_disconnect(self, sid):
        print('[{}] disconnect'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

    
if __name__ == '__main__':
    
    sio = socketio.Server(cors_allowed_origins='*') # CORSのエラーを無視する設定
    sio.register_namespace(MyCustomNamespace('/test')) # 名前空間を設定
    app = socketio.WSGIApp(sio) # wsgiサーバーミドルウェア生成
    eventlet.wsgi.server(eventlet.listen(("localhost", 5000)), app) # wsgiサーバー起動