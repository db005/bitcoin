import pymongo as pm
import pandas as pd


mongoclient = pm.MongoClient("mongodb://localhost:27017/")
db = mongoclient["huobi"]
market = db['market']
candle = db['candle']
timestamp = 1583017197813
mark=[]
cand=[]

market = pd.DataFrame(list(market.find()))
candle = pd.DataFrame(list(candle.find()))
print(market)
markettemp = market[market['Timestamp']==timestamp].index[0]
markettemp = market.iloc[[markettemp]]['Timestamp'].tolist()[0]
print(markettemp)
