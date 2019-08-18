# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 14:53:32 2017

@author: Administrator

tip:
    将键修改为 小写 data03 = {k.lower(): v for k, v in data02.items()}

从wind下载期货行情数据，并存储到本地，同时实现数据更新     
"""

import pandas as pd
import os
import copy
from WindPy import *
import datetime 
#%%
parm = pd.read_excel('F:\\PythonSpace\\Futures_Parameter.xlsx')
dataDir =  'F:\\PythonSpace\\quantiacs\\data'

# 检查是否存在保存数据的文件夹
if not os.path.isdir(dataDir):
    os.mkdir(dataDir)
    print('建立数据文件夹 '+dataDir)
    
# 获取参数列表
code_list = parm['Contract']+'.'+parm['MarketCode']
#%%
def loadData(code,field,startDate,endDate):
    dat = w.wsd(code ,field ,startDate ,endDate ,"TradingCalendar=SHFE") # 期货数据此处指定交易所
    fields01 = [v.lower() for v in dat.Fields]  # 列表推导
    df_data  = pd.DataFrame(dat.Data,index=fields01,columns=dat.Times) # 注意此处Wind下载的数据结构
    df_data  = df_data.T  # 或者 pd.DataFrame.transpose
    return df_data
 
#    将数据存储到pymongo数据库    
#    code = dat.Codes
#    for iIndex in df_data.index:
#        collection = db[iIndex.strftime('%Y-%m-%d')]
#        if collection.find_one({"code": code}) is None:
#            data01 = df_data.loc[iIndex]
#            if not data01.isnull().any() : # df.isnull().any() 判断是否有缺失值
#                data02 = data01.to_dict()
#                data02['code'] = code
#                collection.insert_one(data02) 
def updateData(code_list,startDate = "2000-01-01",endDate = "2017-12-31",field = "open,high,low,close,volume",refresh=False):
    
    w.start()
    
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
            df_data = loadData(iCode,field,startDate,endDate)
            # 代码字符串处理
            df=df_data.dropna(how='all')
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
                df1 = loadData(iCode,fields01,hisData.index[-2],todayDate)
                df = pd.concat([hisData,df1]) 
                df = df.drop_duplicates()   # 删除有重复的数据
                
                if len(diffFields)>=1:
                    df2 = loadData(iCode,fields02,hisData.index[0],todayDate)
                    if df2.shape[0]!=df.shape[0]:
                        print('新增fields数据长度和已有数据长度不匹配')
                    df = pd.concat([df,df2],axis=1)  
                    
                df.to_csv(fileName)
                
#%%        

def get_his_data(marketList=['AG','AU'], dataToLoad=['VOLUME'], refresh=False, beginInSample='2016-01-01', endInSample='2017-12-31',dataDir = 'F:\\PythonSpace\\quantiacs\\data'):
    '''
    返回指定代码的行情数据
    Args:
    Returns:
    '''
    if marketList is None:
        print("warning: no markets supplied")
        return
    
    dataDict = {}
    dataToLoad = set(dataToLoad)
    requiredData = set(['CLOSE'])
    dataToLoad=set(dataToLoad).union(requiredData)
    
    print('Loading Data...')  
    # 设置日期时间范围
    dateRange = pd.date_range(beginInSample,endInSample).strftime('%Y-%m-%d')
    # 读取数据到内存
    for market in marketList:  # Help on class enumerate in module builtins
        marketFile = os.path.join(dataDir, market+'.csv')
        data = pd.read_csv(marketFile)
        data.rename(columns={'Unnamed: 0':'DATE'}, inplace=True)
        data.columns=data.columns.str.upper()  # 将列名都修改为大写格式
        data.set_index('DATE', inplace=True)   # 将日期设为索引 
        
        for j, dataType in enumerate(dataToLoad): 
            if dataType not in dataDict:
                dataDict[dataType] = pd.DataFrame(index=dateRange, columns=marketList)
                
            if dataType != 'DATE' and dataType in data:
                dataDict[dataType][market] = data[dataType]    
            else:
                print(market+' missing ['+dataType+'] field,Please reload data')          
    # 缺失数据处理
    new_dataDict = copy.copy(dataDict)
    for index, dataType in enumerate(dataToLoad):
        if dataType != 'DATE' and dataType in dataDict:
            data = dataDict[dataType].dropna(how='all')
            if 'DATE' not in new_dataDict:
                new_dataDict['DATE'] = list(new_dataDict['CLOSE'].index.values)
                new_dataDict['CODE'] = list(new_dataDict['CLOSE'].columns)
            new_dataDict[dataType] = data.values
            
    print('Done...')  
    return new_dataDict
    


    
                
        
        
            
    
    


                
                
                
                

            
            
            
        
    
    
    
    
    
    






