from huobitrade import setKey
from huobitrade.service import HBWebsocket
setKey('157aa199-f87cd332-mn8ikls4qg-45370', '4e530069-0c505554-fc3971fa-58742')
hb = HBWebsocket(auth=True)  # 可以填入url参数，默认是api.huobi.br.com
@hb.after_auth  # 会再鉴权成功通过之后自动调用
def sub_accounts():
    hb.sub_accounts()

hb.run()  # 开启websocket进程

@hb.register_handle_func('accounts')  # 注册一个处理函数，最好的处理方法应该是实现一个handler
def auth_handle(msg):
    print('auth_handle:', msg)