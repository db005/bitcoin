import logging
import pymongo as pm
from huobi import SubscriptionClient
from huobi.model import *
from huobi.exception.huobiapiexception import HuobiApiException
import time 


logger = logging.getLogger("huobi-client")
logger.setLevel(level=logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

mongoclient = pm.MongoClient("mongodb://localhost:27017/")
db = mongoclient["huobi"]


sub_client = SubscriptionClient()


def callbackMbp(event: 'MbpRequest'):
    # print("Timestamp: " , event.id)
    # print("Channel : " , event.rep)
    mbp = event.data
    # print("seqNum : ", mbp.seqNum)
    # print("prevSeqNum : ", mbp.prevSeqNum)

    col = db["market"]
    
    for i in range(6):
        # print("Bids: " + " price: " + (mbp.bids[i].price) + ", amount: " + (mbp.bids[i].amount))
        mydict = dict()
        mydict = {"Timestamp":(event.id)}
        mydict['direction'] = 'Bids'
        mydict['price'] = (mbp.bids[i].price)
        mydict['amount'] = (mbp.bids[i].amount)
        col.insert_one(mydict)
    
    for i in range(6):
        # print("Asks: " + " price: " + (mbp.asks[i].price) + ", amount: " + (mbp.asks[i].amount))
        mydict = dict()
        mydict = {"Timestamp":(event.id)}
        mydict['direction'] = 'Asks'
        mydict['price'] = (mbp.asks[i].price)
        mydict['amount'] = (mbp.asks[i].amount)
        col.insert_one(mydict)

def errorMbp(e: 'HuobiApiException'):
    print(e.error_code + e.error_message)







def callbackcandle(candlestick_event: 'CandlestickRequest'):

    # print("Symbol: " + (candlestick_event.symbol))
    # print("Timestamp: " + (candlestick_event.timestamp))
    # print("Interval: " + (candlestick_event.interval))
    col = db['candle']

    if len(candlestick_event.data):
        for candlestick in candlestick_event.data:
            mydict = dict()
            candlestick.print_object()
            
            mydict['Timestamp'] = (candlestick_event.timestamp)
            mydict['id'] = (candlestick.id)
            mydict['high'] = (candlestick.high)
            mydict['low'] = (candlestick.low)
            mydict['open'] = (candlestick.open)
            mydict['close'] = (candlestick.close)
            mydict['volume'] = (candlestick.volume)
            mydict['amount'] = (candlestick.amount)
            col.insert_one(mydict)
            print()


def errorcandle(e: 'HuobiApiException'):
    print(e.error_code + e.error_message)

i = 0
#sub_client.request_candlestick_event("btcusdt", CandlestickInterval.MIN1, callback, from_ts_second=None, end_ts_second=None, auto_close=True, error_handler=None)
#sub_client.request_candlestick_event("btcusdt", CandlestickInterval.MIN1, callback, from_ts_second=1569361140, end_ts_second=1569366420)
#sub_client.request_candlestick_event("btcusdt", CandlestickInterval.MIN1, callback, from_ts_second=1569361140, end_ts_second=0)
#sub_client.request_candlestick_event("btcusdt", CandlestickInterval.MIN1, callback, from_ts_second=1569379980)
while True:
    sub_client.request_mbp_event("ethusdt", MbpLevel.MBP150, callbackMbp, errorMbp)
    time.sleep(0.1)