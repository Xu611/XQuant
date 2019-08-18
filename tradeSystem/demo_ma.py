# -*- coding: utf-8 -*-
"""
Created on Wed Nov 01 16:06:06 2017

简单的均线趋势跟踪系统示例

@author: xujw
"""

import numpy as np
import talib
import factor

#def myTradingSystem(OPEN, HIGH, LOW, CLOSE, exposure, equity, settings):
#    ''' This system uses trend following techniques to allocate capital into the desired equities'''

# ------------------------- 修改后的函数-------------------------------
def myTradingSystem(data,settings):
    # 回测使用数据
    CLOSE = data['CLOSE']
    CLEAN = data['CLEAN']
    HIGH  = data['HIGH']
    LOW   = data['LOW']
    VOLUME= data['VOLUME'] 
#    Latest= CLOSE[-1,:]
    #---------------------------我是分割线  -------------------------------
    
    #nMarkets = CLOSE.shape[1]
    
    periodShorter = settings['periodShort']     
    periodLonger  = settings['periodLong']  
    
    filerShort    = settings['filerShort']  
    filerLong     = settings['filerLong']   
    
    riskFactor    = settings['riskFactor']
    
    #---------------------------我是分割线  -------------------------------
    units = CLEAN[-1,:]/CLOSE[-1]
    ATR = np.empty(CLOSE.shape)*np.nan
    
    # 调试语句
    #print(data['DATE'][-1])

        
    for col in range(CLOSE.shape[1]):
        High = HIGH[:,col]
        Low  = LOW[:,col]
        Close= CLOSE[:,col]
        if np.all(~np.isnan(Close) & ~np.isnan(High) & ~np.isnan(Low)):
            ATR[:,col] = talib.ATR(High,Low,Close,timeperiod=periodLonger)
            
    
    
    #---------------------------我是分割线  -------------------------------
    f1=np.nansum(CLOSE[-filerShort:, :], axis=0)/filerShort
    f2=np.nansum(CLOSE[-periodShorter:, :], axis=0)/filerLong
    distance = np.abs(f1/f2-1)
    score = factor.scoreF(distance,settings['level'])
    
    # 选择交易合约
    orderIdx = (score>settings['threshold']) & (VOLUME[-1,:]>10**4) & (CLEAN[-1,:]<2*10**5)
    nMarkets = np.count_nonzero(orderIdx)    # ~np.isnan(Latest)
    
    # Calculate Simple Moving Average (SMA)
    smaLongerPeriod  = np.nansum(CLOSE[-periodLonger:, :], axis=0)/periodLonger
    smaShorterPeriod = np.nansum(CLOSE[-periodShorter:, :], axis=0)/periodShorter

    longEquity  = smaShorterPeriod > smaLongerPeriod
    shortEquity = ~longEquity
    
    
    # 权重分配
    weights = np.zeros(CLOSE.shape[1])
    if nMarkets>0:    
        weights[orderIdx] = riskFactor/nMarkets/(units[orderIdx]*ATR[-1,orderIdx])*CLEAN[-1,orderIdx]
        weights[np.isnan(weights)]=0

#    weights = np.zeros(CLOSE.shape[1])
#    w = 1/nMarkets
#    code = ~np.isnan(Latest)    # 下单合约索引
#    weights[code] = w(code)

    # 下单手数
    amount = settings['budget']*weights/CLEAN[-1,:]
    amount = np.floor(amount)
    amount[np.isnan(amount)] = 0
    # 下单方向
    side = np.zeros(CLOSE.shape[1])
    side[longEquity]  = 1
    side[shortEquity] = -1
    # 持仓信号
    pos = side*amount
#    weights = pos/np.nansum(abs(pos))
    
    return pos,weights,settings      # 对返回值进行修改

def mySettings():
    '''定义交易系统的配置'''

    settings = {}
    # 设置参数
    settings['periodShort'] = 5
    settings['periodLong']  = 20
    settings['level']       = 10
    settings['threshold']   = 3
    settings['filerShort']  = 20
    settings['filerLong']   = 60
    
    settings['riskFactor']  = 0.02
    # Futures Contracts
#    settings['markets']  = ['RB','RU'] 
    settings['markets']  = ['RU','ZN','AU','AG','BU','HC','RB','SN','CU','NI','AL','PB',
                           'A','I','J','JD','JM','L','P','C','Y','CS','M','V',
                           'OI','CF','FG','MA','PP','RM','SR','TA','ZC']
                            
    settings['fields']   = ['CLOSE','HIGH','LOW','VOLUME']
    settings['lookback'] = 130
    settings['budget']   = 10**7
    settings['slippage'] = 0
    settings['beginInSample'] = '2010-01-01'
    settings['endInSample']   = '2017-12-31'
    return settings

# Evaluate trading system defined in current file.

if __name__ == '__main__':
    import __init__
    print('version:'+__init__.__version__)    
    
    import runAlgo
    results = runAlgo.runts(__file__)
