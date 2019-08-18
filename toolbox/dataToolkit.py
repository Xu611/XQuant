# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 14:53:32 2017

@author: Administrator

tip:
    将键修改为 小写 data03 = {k.lower(): v for k, v in data02.items()}

从wind下载期货行情数据，并存储到本地，同时实现数据更新     
"""
#%%
'''
# 更新大商所指数数据

parm  = pd.read_excel('F:\\CommodityIndex\\mfile\\param.xlsx')
month = ['01M','05M','09M']

code_list01 = parm['Contract'].repeat(len(month)) + month*(len(parm['Contract']))
code_list   = code_list01 + '.' + parm['MarketCode'].repeat(len(month)) 
code_list   = sorted(list(set(code_list)))

saveDir   = 'F:\\CommodityIndex\\data'
fields    = 'trade_hiscode,settle,volume,oi,close'
startdate = '2013-01-01'
enddate   = datetime.datetime.now().strftime("%Y-%m-%d")

updateData(code_list,startDate = startdate,endDate = enddate ,field =fields ,dataDir=saveDir)

'''

#%%
'''
更新期货行情数据

path = 'F:\\PythonSpace\\XQuant\\'
parm = pd.read_excel(path+'config\\Futures_Parameter.xlsx')
dataDir = path+'data'

# 检查是否存在保存数据的文件夹
if not os.path.isdir(dataDir):
    os.mkdir(dataDir)
    print('建立数据文件夹 '+dataDir)
    
# 获取参数列表
code_list = parm['Contract']+'.'+parm['MarketCode']

'''

#%%
import pandas as pd
import numpy as np
import os
import copy
from WindPy import *
import datetime 

#%% 
def wstart():
    global w
    w.start()    
#%%
def wLoadData(code,field,startDate,endDate):
    global w
    # 根据合约代码 指定交易所
    code_exc   = code[str.find(code,'.')+1 :]
    exchange01 = "TradingCalendar=SHFE"   # 上海期货交易所
    exchange02 = 'TradingCalendar=CFFEX'  # 中国金融期货交易所
    exchange03 = 'TradingCalendar=DCE'    # 大连商品交易所
    exchange04 = 'TradingCalendar=CZCE'   # 郑州商品交易所
    exchange05 = 'TradingCalendar=SZSE'   # 深圳证券交易所
    
    if str.upper(code_exc) == 'SHF' :
        TradingCalendar = exchange01
    elif str.upper(code_exc) == 'CFE' :
        TradingCalendar = exchange02
    elif str.upper(code_exc) == 'DCE' :
        TradingCalendar = exchange03
    elif str.upper(code_exc) == 'CZC' :
        TradingCalendar = exchange04
    elif str.upper(code_exc) == 'SZ' :
        TradingCalendar = exchange05
    else:
        TradingCalendar = None
    
    if TradingCalendar is not None :
        dat = w.wsd(code ,field ,startDate ,endDate ,TradingCalendar) 
    else:
        dat = w.wsd(code ,field ,startDate ,endDate)
        
    fields01 = [v.lower() for v in dat.Fields]                           # 列表推导，将field统一成小写
    df_data  = pd.DataFrame(dat.Data,index=fields01,columns=dat.Times)   # 注意此处Wind下载的数据结构
    df_data  = df_data.T                                                 # 或者 pd.DataFrame.transpose
    
    return df_data
 
#    将数据存储到pymongo数据库（未完成）    
#    code = dat.Codes
#    for iIndex in df_data.index:
#        collection = db[iIndex.strftime('%Y-%m-%d')]
#        if collection.find_one({"code": code}) is None:
#            data01 = df_data.loc[iIndex]
#            if not data01.isnull().any() : # df.isnull().any() 判断是否有缺失值
#                data02 = data01.to_dict()
#                data02['code'] = code
#                collection.insert_one(data02) 
    
#%%
def updateData(code_list=None,startDate = "2000-01-01",endDate = "2017-12-31",field = "open,high,low,close,volume",dataDir='\\data',refresh=False):
    '''
    更新本地的csv历史数据
    example：
        datapath = 'F:\\MatlabSpace\\AlgoTrading2018\\data\\future_data'
        updateData(code_list,startDate = "2000-01-01",endDate = "2017-12-31",\
        field = "open,high,low,close,volume,trade_hiscode",dataDir=datapath,refresh=True)
    '''
    wstart()
    # 是否存在保存数据的文件夹
    if not os.path.isdir(dataDir):
        os.mkdir(dataDir)
        print('建立数据文件夹 '+dataDir)
        
    if refresh :
        todayDate=datetime.datetime.now().strftime("%Y-%m-%d")  # "%Y-%m-%d %H:%M:%S"
    else:
        todayDate=endDate
        
    for iCode in code_list :
        # 数据文件名+路径
        fileName = dataDir+'\\'+iCode[:str.find(iCode,'.')]+'.csv'
        # 判断行情数据是否存在
        if not os.path.isfile(fileName):
            # 更新数据
            df_data = wLoadData(iCode,field,startDate,endDate)
            # 代码字符串处理
            df=df_data.dropna(how='all')   # 删除整行都缺失的数据
            df.to_csv(fileName)
            print('正在下载'+iCode+'历史数据：'+' from '+startDate+' to '+endDate)
        else:
            with open(fileName) as csvfile:
                hisData = pd.read_csv(csvfile,index_col=0) 
                dates_index = pd.DatetimeIndex(hisData.index).strftime('%Y-%m-%d')  # 修改日期索引格式
                hisData.index = [datetime.datetime.strptime(date,"%Y-%m-%d").date() for date in dates_index] # 修改为datetime.date格式
                
                hisFields = list(hisData.columns.values)
                newFields = field.split(',')
                diffFields= list(set(newFields) - set(hisFields))
      
                fields01 = ",".join(x for x in hisFields)
                fields02 = ",".join(x for x in diffFields)
                
                # 下载更新的数据
                df1 = wLoadData(iCode,fields01,hisData.index[-2],todayDate)
                df = pd.concat([hisData,df1]) 
                df = df.drop_duplicates()   # 删除有重复的数据
                
                if len(diffFields)>=1:
                    df2 = wLoadData(iCode,fields02,hisData.index[0],todayDate)
                    if df2.shape[0]!=df.shape[0] or not all(df2.index == df.index):
                        
                        df3 = wLoadData(iCode,fields01+','+fields02,startDate,todayDate)
                        df3.to_csv(fileName)
                        
                        print(iCode+' 新增fields数据与历史数据长度不匹配，重新下载所有历史数据'+' 更新至：' \
                              +df3.index[-1].strftime('%Y/%m/%d'))

                        continue
                    else:
                        df = pd.concat([df,df2],axis=1)  
                    
                df.to_csv(fileName)
                print(iCode+' 更新至：'+df.index[-1].strftime('%Y/%m/%d'))
    print('Data Update End ...')
#%%        
def get_his_data(marketList=['AG','AU'], dataToLoad=['VOLUME'], refresh=False, beginInSample='2016-01-01', endInSample='2017-12-31',dataDir = 'F:\\PythonSpace\\XQuant\\data'):
    '''
    返回输入指定代码的行情数据，当refresh为true，从API接口更新数据
    Args:
        marketList (list): list of market data to be supplied
        dataToLoad (list): list of financial data types to load
        refresh (bool): boolean value determining whether or not to update the local data from the Data server.
        beginInSample (str): a str in the format of YYYYMMDD defining the begining of the time series
        endInSample (str): a str in the format of YYYYMMDD defining the end of the time series

    Returns:
        dataDict (dict): mapping all data types requested by dataToLoad. The data is returned as a numpy array or list and is ordered by marketList along columns and date along the row.
        
    '''
    if marketList is None:
        print("warning: no markets supplied")
        return
    
    dataDict     = {}
    dataToLoad   = set(dataToLoad)
    requiredData = set(['CLEAN'])   # 修改为清洗后的金融数据
    dataToLoad   = set(dataToLoad).union(requiredData)
    
    print('Loading Data...')  
    
    # 由于期货合约价值具有乘数，及股票数据回测需要复权，增加默认的回测数据方便调用
    paramdir = os.path.dirname(dataDir)+'\\config\\Futures_Parameter.xlsx'
    FuturesParam = pd.read_excel(paramdir)
    Multiplier = FuturesParam['TradingUnits'].tolist()
    Contract   = FuturesParam['Contract'].tolist()
    
    # 设置日期时间范围
    dateRange = pd.date_range(beginInSample,endInSample).strftime('%Y-%m-%d')
    
    # 读取数据到内存
    for market in marketList:  # Help on class enumerate in module builtins
#        print('  load '+ market +' data')
        marketFile = os.path.join(dataDir, market+'.csv')
        data = pd.read_csv(marketFile)
        data.rename(columns={'Unnamed: 0':'DATE'}, inplace=True)
        data.columns=data.columns.str.upper()  # 将列名都修改为大写格式
        data.set_index('DATE', inplace=True)   # 将日期设为索引 
        
        for j, dataType in enumerate(dataToLoad): 
            if dataType not in dataDict:
                # 数据初始化
                dataDict[dataType] = pd.DataFrame(index=dateRange, columns=marketList) 
            if dataType != 'DATE' and dataType in data:
                dataDict[dataType][market] = data[dataType]   
            elif dataType == 'CLEAN' and market in Contract:
                dataDict[dataType][market] = data['CLOSE']*Multiplier[Contract.index(market)]
            elif dataType != 'DATE' and dataType not in data :
                print(market+' missing ['+dataType+'] field,Please reload data') 
                
    # 缺失数据处理
    new_dataDict = copy.copy(dataDict)
    
    k=-1
    cleanIdx = np.ones((new_dataDict['CLEAN'].shape[0], len(new_dataDict)), dtype=bool)
    for key, value in new_dataDict.items():
        k+=1
        cleanIdx[:,k] = pd.isnull(value).all(axis=1).tolist()
    selectIdx = ~cleanIdx.all(axis=1)
    
    outDict = {}
    for key, value in new_dataDict.items():
        # 对缺失数据使用前一值进行填充
        value1 = value[selectIdx].fillna(method='ffill')
        outDict[key] = value1.values
        if  'DATE' not in outDict:
            outDict['DATE'] = value1.index.values.tolist()
        if  'CODE' not in new_dataDict:
            outDict['CODE'] = value1.columns.values.tolist()
        
        
#    for index, dataType in enumerate(dataToLoad):
#        if dataType != 'DATE' and dataType in dataDict:
#            data_dropna = dataDict[dataType].dropna(how='all')
#            if 'DATE' not in new_dataDict:
#                new_dataDict['DATE'] = list(data_dropna.index.values)
#                new_dataDict['CODE'] = list(data_dropna.columns)
#            new_dataDict[dataType] = data_dropna.values
            
    print('Done...')  
    return outDict
        
#%%
def stats(equityCurve):
    ''' calculates trading system statistics
        计算风险指标
    Calculates and returns a dict containing the following statistics
    - sharpe ratio
    - sortino ratio
    - annualized returns
    - annualized volatility
    - maximum drawdown
        - the dates at which the drawdown begins and ends
    - the MAR ratio
    - the maximum time below the peak value
        - the dates at which the max time off peak begin and end

    Args:
        equityCurve (list): the equity curve of the evaluated trading system

    Returns:
        statistics (dict): a dict mapping keys to corresponding trading system statistics (sharpe ratio, sortino ration, max drawdown...)

    Copyright Quantiacs LLC - March 2015

    '''
    returns = (equityCurve[1:]-equityCurve[:-1])/equityCurve[:-1]

    volaDaily=np.std(returns)
    volaYearly=np.sqrt(252)*volaDaily

    index=np.cumprod(1+returns)
    indexEnd=index[-1]

    returnDaily = np.exp(np.log(indexEnd)/returns.shape[0])-1
    returnYearly = (1+returnDaily)**252-1
    sharpeRatio = returnYearly / volaYearly

    downsideReturns = returns.copy()
    downsideReturns[downsideReturns > 0]= 0
    downsideVola = np.std(downsideReturns)
    downsideVolaYearly = downsideVola *np.sqrt(252)

    sortino = returnYearly / downsideVolaYearly

    highCurve = equityCurve.copy()

    testarray = np.ones((1,len(highCurve)))
    test = np.array_equal(highCurve,testarray[0])

    if test:
        mX      = np.NaN
        mIx     = np.NaN
        maxDD   = np.NaN
        mar     = np.NaN
        maxTimeOffPeak = np.NaN
        mtopStart = np.NaN
        mtopEnd = np.NaN
    else:
        for k in range(len(highCurve)-1):
            if highCurve[k+1] < highCurve[k]:
                highCurve[k+1] = highCurve[k]

        underwater = equityCurve / highCurve
        mi = np.min(underwater)
        mIx = np.argmin(underwater)
        maxDD = 1 - mi
        mX= np.where(highCurve[0:mIx-1] == np.max(highCurve[0:mIx-1]))
        #        highList = highCurve.copy()
        #        highList.tolist()
        #        mX= highList[0:mIx].index(np.max(highList[0:mIx]))
        mX=mX[0][0]
        mar=returnYearly / maxDD

        mToP = equityCurve < highCurve
        mToP = np.insert(mToP, [0,len(mToP)],False)
        mToPdiff=np.diff(mToP.astype('int'))
        ixStart   = np.where(mToPdiff==1)[0]
        ixEnd     = np.where(mToPdiff==-1)[0]

        offPeak         = ixEnd - ixStart
        if len(offPeak) > 0:
            maxTimeOffPeak  = np.max(offPeak)
            topIx           = np.argmax(offPeak)
        else:
            maxTimeOffPeak = 0
            topIx          = np.zeros(0)

        if np.not_equal(np.size(topIx),0):
            mtopStart= ixStart[topIx]-2
            mtopEnd= ixEnd[topIx]-1

        else:
            mtopStart = np.NaN
            mtopEnd = np.NaN
            maxTimeOffPeak = np.NaN

    statistics={}
    statistics['sharpe']              = sharpeRatio
    statistics['sortino']             = sortino
    statistics['returnYearly']        = returnYearly
    statistics['volaYearly']          = volaYearly
    statistics['maxDD']               = maxDD
    statistics['maxDDBegin']          = mX
    statistics['maxDDEnd']            = mIx
    statistics['mar']                 = mar
    statistics['maxTimeOffPeak']      = maxTimeOffPeak
    statistics['maxTimeOffPeakBegin'] = mtopStart
    statistics['maxTimeOffPeakEnd']   = mtopEnd

    return statistics