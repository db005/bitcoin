import pymongo as pm
import pandas as pd
import time

SLEEP = 0.1
FEE = 0.0007

class environment:
    def __init__(self , eth, usdt, virtual, timestamp=0 ):
        mongoclient = pm.MongoClient("mongodb://localhost:27017/")
        db = mongoclient["huobi"]
        market = db['market']
        self.market = pd.DataFrame(list(market.find()))
        if timestamp!=0:
            self.timestamp = timestamp
        else:
            tm = self.market['Timestamp'].min()

            self.marketindex = self.market[self.market['Timestamp']==tm].index[0]
            self.marketindex += 64 * 12 * 600 - 1
            self.timestamp = tm
            
        self.ethprice = 0
        self.ethamount = eth
        self.usdtamount = usdt
        self.virtual = virtual
        self.reward0 = 0
        self.reward1 = 0
        self.reward = 0
        self.amount = 0
    
    def reset(self):
        while self.market.shape[0]<self.marketindex+12:
            time.sleep(SLEEP)
            mongoclient = pm.MongoClient("mongodb://localhost:27017/")
            db = mongoclient["huobi"]
            market = db['market']
            self.market = pd.DataFrame(list(market.find()))
        markettemp = self.market[self.market['Timestamp']==self.market.iloc[[self.marketindex]]['Timestamp'].tolist()[0]]
        
        while markettemp.shape[0]!=12:
            self.marketindex += 12
            if self.market.shape[0]-self.marketindex>=12:
                markettemp = self.market[self.market['Timestamp']==self.market.iloc[[self.marketindex]]['Timestamp'].tolist()[0]]
                
            else:
                time.sleep(SLEEP)
                mongoclient = pm.MongoClient("mongodb://localhost:27017/")
                db = mongoclient["huobi"]
                market = db['market']
                self.market = pd.DataFrame(list(market.find()))
            
        marketprice = markettemp['price'].tolist()
        marketamount = markettemp['amount'].tolist()
        
        candleindex = self.marketindex
        candle = []
        for i in range(64):
            candle.append(self.market['price'].get(candleindex))
            candle.append(self.market['price'].get(candleindex - 6))
            candleindex -= 600 * 12

        # print(marketprice+marketamount+candle)
        self.marketindex += 12
        return marketprice+marketamount+candle+[self.usdtamount,self.ethamount]
    
    def step(self,action):
        state = self.reset()
        # print("Now:",time.asctime(time.localtime(time.time())),"From:",time.asctime(time.localtime(self.market.iloc[[self.marketindex-12]]['Timestamp'].tolist()[0]/1000)),state)
        done = False
        if len(state)<154:
            done = True
            
            for i in range(154-len(state)):
                state.append(0)
            return state,self.reward,done
        self.ethprice = state[0]
        market = []
        for i in range(12):
            market.append(state[i])
            market.append(state[i+12])
        if self.virtual:
            if action[1]>0:
                if action[0]>0.9:
                    action[0]=1
                if action[0]>0.01:
                    usdtamount = self.usdtamount*action[0]
                    usdttemp = usdtamount
                    if usdtamount >= 5:
                        for i in range(6):
                            if (usdtamount - market[2*i]*market[2*i+1]) >= 0:
                                self.ethamount += market[2*i+1]
                                self.usdtamount -= market[2*i]*market[2*i+1]
                                usdtamount -= market[2*i]*market[2*i+1]
                            else:
                                self.ethamount += usdtamount/market[2*i]
                                self.usdtamount -= usdtamount
                                break
                        if self.usdtamount>=FEE*usdttemp:
                            self.usdtamount -= FEE*usdttemp
                if action[0]<-0.9:
                    action[0]=-1
                if action[0]<-0.01:
                    ethamount = self.ethamount*(-1)*action[0]
                    ethtemp = ethamount
                    if ethamount>=0.01:
                        for i in range(6):
                            if (ethamount - market[2*i+12]) >= 0:
                                self.ethamount -= market[2*i+12]
                                self.usdtamount += market[2*i+12]*market[2*i+13]
                                ethamount -= market[2*i+12]
                            else:
                                self.ethamount -= ethamount
                                self.usdtamount += ethamount*market[2*i+12]
                                break
                        if self.ethamount>=FEE*ethtemp:
                            self.ethamount -= FEE*ethtemp
        
        if self.reward1 == 0:
            self.reward0 = self.ethamount * self.ethprice + self.usdtamount
        else:
            self.reward0 = self.reward1

        self.reward1 = self.ethamount * self.ethprice + self.usdtamount
        self.reward = self.reward1 - self.reward0
        if self.reward1<8:
            self.usdtamount = 200
            self.reward = 0
        print(self.reward1, self.reward)
        return state,self.reward,done


