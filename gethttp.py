from flask import Flask
from rl import train
from multiprocessing import Process, Queue
import asyncio
app = Flask(__name__)

queue=Queue()
@app.route('/')
def hello_world():
    
    return queue.get(True)

def apprun():
    # 监听用户请求
    # 如果有用户请求到来，则执行app的__call__方法
    # app.__call__
    app.run(port=80)

if __name__ == '__main__':   
    main = Process(target=apprun)
    rl = Process(target=train,args=(queue,))
    main.start()
    main.join()
    rl.start()
    rl.join()
