# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 10:22:52 2017

基于python的回测框架
@author: quantiacs , @reviser：xujw  

Edition：
    1. Python 3.6 or Python2.7 兼容
    2. 在数据中新增CLEAN 数据，使用清洗后的数据来用于回测
    3. 多空持仓手数用颜色区分，代替数值区分
""" 
#python2.7 导入整数除法
#from __future__ import division

import traceback
import imp               # 实现自定义的 import 

import time              # 日期时间处理
import os                # os操作文件目录  
import os.path
import sys               # sys处理运行系统环境
#import inspect

from copy import deepcopy
import numpy as np
np.seterr(divide='ignore', invalid='ignore')   # 处理RuntimeWarning: invalid value encountered in true_divide

# 2018-01-09 添加独立的数据管理模块，后期将把loadData从当前模块移除
# 2018-03-08 添加独立的GUI模块plotts，把当前模块画图函数移除。将stats函数移动到dataToolkit中
from dataToolkit import get_his_data as loadData    # 导入同目录下的模块
from dataToolkit import stats    
from performance import plot as plotts
#from plot import plotts
#%%
def runts(tradingSystem, plotEquity=True, reloadData=False, state={}, sourceData='tickerData'):
    '''  回测交易系统：backtests a trading system.
    Input:
        tradingSystem: '/samplesystems/demo_ma.py'
    Example:
    s = runts('tsName') evaluates the trading system specified in string tsName, and stores the result in struct s.

    Args:
        tsName (str): Specifies the trading system to be backtested
        plotEquity (bool, optional): Show the equity curve plot after the evaluation
        reloadData (bool,optional): Force reload of market data.
        state (dict, optional): State information to resume computation of an existing backtest (for live evaluation on Quantiacs servers). State needs to be of the same form as ret.
    
    Vars:
        CLOSE: the last price of the session
        exposure: the realized quantities of your trading system, or all the trading positions you take
        equity: cumulative trading performance in each market, reflects gains and losses
    
    Returns:
        a dict mapping keys to the relevant backesting information: trading system name, system equity, trading dates, market exposure, market equity, the errorlog, the run time, the system's statistics, and the evaluation date.

        keys and description:
            'tsName' (str):           Name of the trading system, same as tsName
            'fundDate' (int):         All dates of the backtest in the format YYYYMMDD
            'fundEquity' (float):     Equity curve for the fund (collection of all markets)
            'returns' (float):        Marketwise returns of trading system
            'marketEquity' (float):   Equity curves for each market in the fund
            'marketExposure' (float): Collection of the returns p of the trading system function. Equivalent to the percent expsoure of each market in the fund. Normalized between -1 and 1
            'settings' (dict):        The settings of the trading system as defined in file tsName
            'errorLog' (list):        list of strings with error messages
            'runtime' (float):        Runtime of the evaluation in seconds
            'stats' (dict):           Performance numbers of the backtest
            'evalDate' (datetime):    Last market data present in the backtest
    '''

    errorlog=[]   # 错误日志
    ret={}
    
    # 获取  TSobject settings tsName
    if type(tradingSystem) is str:
        tradingSystem = tradingSystem.replace('\\', '/')
 
    if str(type(tradingSystem)) == "<type 'classobj'>" or str(type(tradingSystem)) == "<type 'type'>":
        TSobject = tradingSystem()
        settings = TSobject.mySettings()
        tsName   = str(tradingSystem)

    elif str(type(tradingSystem)) == "<type 'instance'>" or str(type(tradingSystem)) == "<type 'module'>":
        TSobject = tradingSystem
        settings = TSobject.mySettings()
        tsName   = str(tradingSystem)

    elif os.path.isfile(tradingSystem):
        
        filePath = str(tradingSystem)
        tsFolder, tsName = os.path.split(filePath)

        try:
            TSobject = imp.load_source('tradingSystemModule', filePath)
        except Exception as e:
            print ('Error loading trading system')
            print (str(e))
            print (traceback.format_exc())
            return

        try:
            settings = TSobject.mySettings()
        except Exception as e:
            print ("Unable to load settings. Please ensure your settings definition is correct")
            print (str(e))
            print (traceback.format_exc())
            return

    else:
        print("Please input your trading system's file path or a callable object.")
        return
    #---------
    if isinstance(state, dict):
        if 'save' not in state:
            state['save']=False
        if 'resume' not in state:
            state['resume']=False
        if 'runtimeInterrupt' not in state:
            state['runtimeInterrupt'] = False
    else:
        print( 'state variable is not a dict' )
    #---------
    # 获取期货的索引 get boolean index of futures(Delete)

    # 获取数据指标 get data fields and extract them.
#    requiredData = set(['DATE','OPEN','HIGH','LOW','CLOSE','P','RINFO','p'])
    requiredData = set(['CLOSE'])
    dataToLoad = requiredData
    
    # 从settings中更新所需要下载的指标，区别于之前读取myTradingSystem参数形式
    dataToLoad.update(settings['fields'])
    
    # 获取函数参数的名称和默认值
#    tsArgs = inspect.getargspec(TSobject.myTradingSystem)
#    tsArgs = tsArgs[0]    # tsArgs : ['DATE','OPEN','CLOSE',...]
#    tsDataToLoad = [item for index, item in enumerate(tsArgs) if item.isupper()] # tsDataToLoad: ['DATE','OPEN',...,'VOL']

#    dataToLoad.update(tsDataToLoad)

    # 定义全局变量
    global settingsCache
    global dataCache
    
    # 下载历史数据 存放到 dataDict 并创建深拷贝的副本
    if 'settingsCache' not in globals() or settingsCache != settings:
        if 'beginInSample' in settings and 'endInSample' in settings:
            dataDict=loadData(settings['markets'],dataToLoad,reloadData, beginInSample = settings['beginInSample'], endInSample = settings['endInSample'])
        elif 'beginInSample' in settings and 'endInSample' not in settings:
            dataDict=loadData(settings['markets'],dataToLoad,reloadData, settings['beginInSample'])
        elif 'endInSample' in settings and 'beginInSample' not in settings:
            dataDict=loadData(settings['markets'],dataToLoad,reloadData, endInSample = settings['endInSample'])
        else:
            dataDict=loadData(settings['markets'],dataToLoad,reloadData)

        dataCache = deepcopy(dataDict)
        settingsCache = deepcopy(settings)

    else:
        print('copying data from cache')
        settings = deepcopy(settingsCache)
        dataDict = deepcopy(dataCache)

    print('Evaluating Trading System Now')
    
    # 初始化变量 Initialize variables 
    nMarkets=len(settings['markets'])
    endLoop =len(dataDict['DATE'])
    
    # RIX获取 RINFO 非零索引
    if 'RINFO' in dataDict:    # Debug :True
        #Rix= dataDict['RINFO'] != 0
        pass
    else:
        dataDict['RINFO'] = np.zeros(np.shape(dataDict['CLEAN']))
        #Rix = np.zeros(np.shape(dataDict['CLEAN']))
   
    # --- 新增 20171129 ---
    dataDict['netMargin'] = np.zeros((endLoop,nMarkets))
    dataDict['yields'] = np.zeros((endLoop,nMarkets))  
    runInfo = {}
    # --- 新增 20171129 ---
    dataDict['exposure'] = np.zeros((endLoop,nMarkets))
    dataDict['equity'] = np.ones((endLoop,nMarkets))
    dataDict['fundEquity'] = np.ones((endLoop,1))
    
    # --- 新增 20180314 ---
    dataDict['weight'] = np.zeros((endLoop,nMarkets))
    
#    realizedP = np.zeros((endLoop, nMarkets))
#    returns = np.zeros((endLoop, nMarkets))
#
#    sessionReturnTemp=np.append( np.empty((1,nMarkets))*np.nan,(( dataDict['CLOSE'][1:,:]- dataDict['OPEN'][1:,:]) / dataDict['CLOSE'][0:-1,:] ), axis =0 ).copy()
#    sessionReturn=np.nan_to_num( fillnans(sessionReturnTemp) )
#    
#    gapsTemp=np.append(np.empty((1,nMarkets))*np.nan, (dataDict['OPEN'][1:,:]- dataDict['CLOSE'][:-1,:]-dataDict['RINFO'][1:,:].astype(float)) / dataDict['CLOSE'][:-1:],axis=0)
#    gaps=np.nan_to_num(fillnans(gapsTemp))

    # 检查是否指定了默认滑点 
    if 'slippage' not in settings:
        settings['slippage'] = 0  # 可选,默认修改为 0
        
#    slippageTemp = np.append(np.empty((1,nMarkets))*np.nan, ((dataDict['HIGH'][1:,:] - dataDict['LOW'][1:,:]) / dataDict['CLOSE'][:-1,:] ), axis=0) * settings['slippage']
#    SLIPPAGE = np.nan_to_num(fillnans(slippageTemp))

    # 调用循环开始的索引，默认从第2个数据开始回测，回溯期为1
    if 'lookback' not in settings:
        startLoop=2
        settings['lookback']=1
    else:
        startLoop=settings['lookback']-1

    # Server evaluation --- resumes for new day.(Delete)

    # 计时器
    t0= time.time()

    # -----------------交易日循环 Loop through trading days --------------------
    for t in range(startLoop,endLoop):

        todaysPos = dataDict['exposure'][t-1,:] 
        todaysW   = dataDict['weight'][t-1,:]           # todaysW中的值都是非负数
        todaysSide= np.sign(todaysPos)
        
        deltaPri  = dataDict['CLEAN'][t,:] - dataDict['CLEAN'][t-1,:]
        deltaRet  = np.log(dataDict['CLEAN'][t,:]/dataDict['CLEAN'][t-1,:])  # 注意取对数收益率

#        posValue  = todaysPos*dataDict['CLEAN'][t-1,:]  # 计算持仓市值
#        posValue[np.isnan(posValue)] = 0                # 缺失值处理
#        posWeight = posValue/np.sum(np.abs(posValue))   # 注意符号处理，取绝对值，可能存在未空值或0的情况       
            
        realizedProfit=deltaPri*todaysPos   
        realizedProfit[np.isnan(realizedProfit)]=0
        realizedRet=deltaRet*todaysW*todaysSide
        realizedRet[np.isnan(realizedRet)]=0

        dataDict['netMargin'][t,:] = realizedProfit
        dataDict['yields'][t,:] = realizedRet
        # ---------------------- xujw edit code-----------------------
        dataDict['equity'][t,:] = dataDict['equity'][t-1,:]*(1+ realizedRet)
        dataDict['fundEquity'][t] = (dataDict['fundEquity'][t-1] * (1+np.sum(realizedRet)))
        
        # 可调用的回测结果
        runInfo['marketPosition']=todaysSide
        
        # Roll futures contracts.(Delete)
        
        # ---------------------- xujw edit code-----------------------
        # 减少函数输入参数的数量，第一个为需要的数据，第二个为回测参数
        try:
            dataCall = {}
            for idx in settings['fields']:
                dataCall[idx] = dataDict[idx][t- settings['lookback'] +1:t+1].copy()
                
            dataCall['CLEAN']   = dataDict['CLEAN'][t- settings['lookback'] +1:t+1].copy()
            dataCall['runInfo'] = runInfo
            dataCall['DATE']    = dataDict['DATE'][t- settings['lookback'] +1:t+1].copy()
            argList = [dataCall,settings]    
                             
            # 20180314 修改函数返回值 新增 weights,用来计算组合收益率
            position, weights ,settings = TSobject.myTradingSystem(*argList)
        except:
            print ('Error evaluating trading system')
            print (sys.exc_info()[0])
            print (traceback.format_exc())
            errorlog.append(str(dataDict['DATE'][t])+ ': ' + str(sys.exc_info()[0]))
            dataDict['equity'][t:,:] = np.tile(dataDict['equity'][t,:],(endLoop-t,1))
            return
            
#        try:
#            argList= []
#
#            for index in range(len(tsArgs)):
#                if tsArgs[index]=='settings':
#                    argList.append(settings)
#                elif tsArgs[index]=='runInfo':   # 201712-11 新增回测信息调用
#                    argList.append(runInfo)                     
#                elif tsArgs[index] == 'self':
#                    continue
#                else:
#                    argList.append(dataDict[tsArgs[index]][t- settings['lookback'] +1:t+1].copy())
#                    
#            # 策略执行
#            position, settings= TSobject.myTradingSystem(*argList)
#            
#        except:
#            print ('Error evaluating trading system')
#            print (sys.exc_info()[0])
#            print (traceback.format_exc())
#            errorlog.append(str(dataDict['DATE'][t])+ ': ' + str(sys.exc_info()[0]))
#            dataDict['equity'][t:,:] = np.tile(dataDict['equity'][t,:],(endLoop-t,1))
#            return
        
        position[np.isnan(position)] = 0
        position = np.real(position)      # Return the real part of the elements of the array
        dataDict['exposure'][t,:] = position.copy()
        
        # --- 20180314 新增 ---
        dataDict['weight'][t,:] = weights.copy()
        
        # 如果运行超过指定时间，运行终止
        t1=time.time()
        runtime = t1-t0
        if runtime > 300 and state['runtimeInterrupt']:
            errorlog.append('Evaluation stopped: Runtime exceeds 5 minutes.')
            break
    # ------------------------ loop end --------------------------------------
            
    if 'budget' in settings:
        fundequity = dataDict['fundEquity'][(settings['lookback']-1):,:] * settings['budget']
    else:
        fundequity = dataDict['fundEquity'][(settings['lookback']-1):,:]

    marketRets = np.float64(dataDict['CLOSE'][1:,:] - dataDict['CLOSE'][:-1,:] - dataDict['RINFO'][1:,:])/dataDict['CLOSE'][:-1,:]
    marketRets = fillnans(marketRets)
    marketRets[np.isnan(marketRets)] = 0
    marketRets = marketRets.tolist()
    a = np.zeros((1,nMarkets))
    a = a.tolist()
    marketRets = a + marketRets

    ret['returns'] = np.nan_to_num(dataDict['yields']).tolist()
    
    statistics = stats(fundequity)  
    

    ret['tsName']=tsName
    ret['fundDate']=dataDict['DATE']                       # ret['fundDate']=dataDict['DATE'].tolist() 变量类型目前是list不进行转换
    ret['fundEquity']=dataDict['fundEquity'].tolist()      # dataDict['fundEquity'].tolist()
    ret['marketEquity']=dataDict['equity'].tolist()
    ret['marketExposure']=dataDict['exposure'].tolist()
    ret['errorLog']=errorlog
    ret['runtime']=runtime
    ret['stats']=statistics
    ret['settings']=settings
    ret['evalDate']=dataDict['DATE'][t]

    # 测试语句
    ret['data'] = dataDict   
    ret['state']= state
    
    #--------------------------------------------------------------------------
    if errorlog:
        print ('Error: {}'.format(errorlog))

    if plotEquity:
        print('Show The Performance...')
        #returns = plotts(tradingSystem, fundequity,dataDict['equity'], dataDict['exposure'], settings, dataDict['DATE'][settings['lookback']-1:], statistics,ret['returns'],marketRets)
#        plotts(tradingSystem, fundequity,dataDict['equity'], dataDict['exposure'], settings, dataDict['DATE'][settings['lookback']-1:], statistics,ret['returns'],marketRets)
        plotts(ret)
        
    return ret

#%%    return True
def computeFees(equityCurve, managementFee,performanceFee):
    ''' computes equity curve after fees
        计算权益曲线
    Args:
        equityCurve (list, numpy array) : a column vector of daily fund values
        managementFee (float) : the management fee charged to the investor (a portion of the AUM charged yearly)
        performanceFee (float) : the performance fee charged to the investor (the portion of the difference between a new high and the most recent high, charged daily)

    Returns:
        returns an equity curve with the fees subtracted.  (does not include the effect of fees on equity lot size)

    '''
    returns = (np.array(equityCurve[1:])-np.array(equityCurve[:-1]))/np.array(equityCurve[:-1])
    ret = np.append(0,returns)

    tradeDays = ret > 0
    firstTradeDayRow = np.where(tradeDays is True)
    firstTradeDay = firstTradeDayRow[0][0]

    manFeeIx = np.zeros(np.shape(ret),dtype=bool)
    manFeeIx[firstTradeDay:] = 1
    ret[manFeeIx] = ret[manFeeIx] - managementFee/252

    ret = 1 + ret
    r = np.ndarray((0,0))
    high = 1
    last = 1
    pFee = np.zeros(np.shape(ret))
    mFee = np.zeros(np.shape(ret))

    for k in range(len(ret)):
        mFee[k] = last * managementFee/252 * equityCurve[0][0]
        if last * ret[k] > high:
            iFix = high / last
            iPerf = ret[k] / iFix
            pFee[k] = (iPerf - 1) * performanceFee * iFix * equityCurve[0][0]
            iPerf = 1 + (iPerf - 1) * (1-performanceFee)
            r=np.append(r,iPerf * iFix)
        else:
            r=np.append(r,ret[k])
        if np.size(r)>0:
            last = r[-1] * last
        if last > high:
            high = last

    out = np.cumprod(r)
    out = out * equityCurve[0]

    return out

#%%
def fillnans(inArr):
    ''' fills in (column-wise)value gaps with the most recent non-nan value.

    fills in value gaps with the most recent non-nan value.
    Leading nan's remain in place. The gaps are filled in only after the first non-nan entry.

    Args:
        inArr (list, numpy array)
    Returns:
        returns an array of the same size as inArr with the nan-values replaced by the most recent non-nan entry.

    '''
    inArr  = inArr.astype(float)
    nanPos = np.where(np.isnan(inArr))
    nanRow = nanPos[0]
    nanCol = nanPos[1]
    myArr  = inArr.copy()
    for i in range(len(nanRow)):
        if nanRow[i] > 0:
            myArr[nanRow[i], nanCol[i]] = myArr[nanRow[i]-1, nanCol[i]]
    return myArr

#%%
def fillwith(field, lookup):
    ''' replaces nan entries of field, with values of lookup.

    Args:
        field (list, numpy array) : array whose nan-values are to be replaced
        lookup (list, numpy array) : array to copy values for placement in field

    Returns:
        returns array with nan-values replaced by entries in lookup.
    '''
    out = field.astype(float)
    nanPos= np.where(np.isnan(out))
    nanRow=nanPos[0]
    nanCol=nanPos[1]

    for i in range(len(nanRow)):
        out[nanRow[i],nanCol[i]] = lookup[nanRow[i]-1,nanCol[i]]

    return out

#%%
def ismember(a, b):
    bIndex = {}
    for item, elt in enumerate(b):
        if elt not in bIndex:
            bIndex[elt] = item
    return [bIndex.get(item, None) for item in a]