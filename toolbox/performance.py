# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 14:47:53 2018

@author: xujw
"""

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2TkAgg
from matplotlib import style
import matplotlib.pyplot as plt
#from matplotlib.figure import Figure # 此函数不能使用

#from numpy import arange, sin, pi
import numpy as np
import datetime

#%%
#import pdb
#pdb.set_trace()
#%% 根据Python版本导入GUI图形库
try:
    import Tkinter as tk
#    import tkFont
    import ttk
except ImportError:  # Python 3
    import tkinter as tk
#    import tkinter.font as tkFont
    import tkinter.ttk as ttk

#%%
def plot(results):
    settings = results['settings']
    markets  = results['settings']['markets']
    equity   = results['data']['equity']
    exposure = results['data']['exposure']
    DATE     = results['data']['DATE']
    returns  = results['returns']
    marketReturns = results['marketEquity']
    fundEquity    = np.array(results['fundEquity'])
    
    DATEord=[]
    lng=len(DATE)
    for i in range(lng):
        #DATEord.append(datetime.datetime.strptime(str(DATE[i]),'%Y%m%d')) # 旧代码 
        DATEord.append(datetime.datetime.strptime(DATE[i],'%Y-%m-%d'))  
    t = np.array(DATEord)  
    
    if 'fundEquity' not in markets:
        markets.insert(0,'fundEquity') 
        
    #--------------------------------------------------------------------------
    equityList = np.transpose(equity)
    
    Long = np.transpose(np.array(exposure))
    Long[Long<0] = 0
    Long = Long[:,(settings['lookback']-2):-1]     # Market Exposure lagged by one day

    Short = - np.transpose(np.array(exposure))
    Short[Short<0] = 0
    Short = Short[:,(settings['lookback']-2):-1]


    returnsList = np.transpose(np.array(returns))

    returnLong = np.transpose(np.array(exposure))
    returnLong[returnLong<0] = 0
    returnLong[returnLong > 0] = 1
    returnLong = np.multiply(returnLong[:,(settings['lookback']-2):-1],returnsList[:,(settings['lookback']-1):])      # y values for Long Only Equity Curve


    returnShort = - np.transpose(np.array(exposure))
    returnShort[returnShort<0] = 0
    returnShort[returnShort > 0] = 1
    returnShort = np.multiply(returnShort[:,(settings['lookback']-2):-1],returnsList[:,(settings['lookback']-1):])    # y values for Short Only Equity Curve

    marketRet = np.transpose(np.array(marketReturns))
    marketRet = marketRet[:,(settings['lookback']-1):]
    equityList = equityList[:,(settings['lookback']-1):]    # y values for all individual markets
    
    #--------------------------------------------------------------------------
    def subplt(t,Exposure,Equity):
        
#        Subplot_Equity = plt.subplot2grid((8,8), (0,0), colspan = 6, rowspan = 6)
#        Subplot_Exposure = plt.subplot2grid((8,8), (6,0), colspan = 6, rowspan = 2, sharex = Subplot_Equity)
        
        Subplot_Equity = plt.subplot2grid((6,3), (0,0), colspan = 3, rowspan = 4)
        Subplot_Exposure = plt.subplot2grid((6,3), (4,0), colspan = 3, rowspan = 2, sharex = Subplot_Equity)
        
        Subplot_Equity.plot(t,Equity,'b',linewidth=0.5)
        Subplot_Equity.set_yscale('log')
        Subplot_Equity.set_ylabel('Performance (Logarithmic)')
        
        Long  = np.transpose(np.array(Exposure))
        Long[Long<0]   = np.nan      # 0 修改为 nan 便于数据可视化      

        Short = - np.transpose(np.array(Exposure))
        Short[Short<0] = np.nan

        Subplot_Exposure.plot(t,Long,'c',linewidth=0.5, label = 'Long')
        Subplot_Exposure.plot(t,Short,'g',linewidth=0.5, label = 'Short')
        
#        Subplot_Exposure.bar(np.arange(len(t)),Long)
#        Subplot_Exposure.bar(np.arange(len(t)),Short) 
        
        statistics=stats(Equity)
        Subplot_Equity.plot(t[statistics['maxDDBegin']:statistics['maxDDEnd']+1],
                            Equity[statistics['maxDDBegin']:statistics['maxDDEnd']+1],color='red',linewidth=0.5, label = 'Max Drawdown')
        if not(np.isnan(statistics['maxTimeOffPeakBegin'])) and not(np.isnan(statistics['maxTimeOffPeak'])):
            Subplot_Equity.plot(t[(statistics['maxTimeOffPeakBegin']+1):(statistics['maxTimeOffPeakBegin']+statistics['maxTimeOffPeak']+2)],
                                  Equity[statistics['maxTimeOffPeakBegin']+1]*np.ones((statistics['maxTimeOffPeak']+1)),'r--',linewidth=2, label = 'Max Time Off Peak')
        
        Subplot_Equity.autoscale(tight=True)
        Subplot_Equity.get_xaxis().set_visible(False)
        Subplot_Equity.legend(bbox_to_anchor=(1.01, 1), loc='lower right', borderaxespad=0.)
        
#        Subplot_Equity.set_title('Trading Performance of %s' %settings['markets'][indx_TradingPerf])
        Subplot_Exposure.autoscale(tight=True)
        Subplot_Exposure.set_ylabel('Long/Short')
        Subplot_Exposure.set_xlabel('Year')
        Subplot_Exposure.legend(bbox_to_anchor=(1.01, 0.7), loc=2, borderaxespad=0.)
        
        plt.gcf().canvas.draw()
        
    def plot2(t,mRet,plotStat=False):
#        plt.clf()
        MarketReturns = plt.subplot2grid((1,1), (0,0), colspan = 1, rowspan = 1)
        MarketReturns.plot(t,mRet,'b',linewidth=0.5)
        
        if plotStat:
            statistics=stats(mRet)
            MarketReturns.plot(t[statistics['maxDDBegin']:statistics['maxDDEnd']+1],
                            mRet[statistics['maxDDBegin']:statistics['maxDDEnd']+1],color='red',linewidth=0.5, label = 'Max Drawdown')

            MarketReturns.plot(t[(statistics['maxTimeOffPeakBegin']+1):(statistics['maxTimeOffPeakBegin']+statistics['maxTimeOffPeak']+2)],
                                  mRet[statistics['maxTimeOffPeakBegin']+1]*np.ones((statistics['maxTimeOffPeak']+1)),'r--',linewidth=2, label = 'Max Time Off Peak')
            MarketReturns.legend(bbox_to_anchor=(1.01, 1), loc='lower right', borderaxespad=0.)
        MarketReturns.set_ylabel('Market Returns')
        MarketReturns.autoscale(tight=True)
        MarketReturns.set_xlabel('Date')

        plt.gcf().canvas.draw()
        
    #--------------------------------------------------------------------------
    def newselection1(event):
        value1 = my_box1.get()
        value2 = my_box2.get()
        if value1=='Return' and value2 == 'fundEquity' :
            plot2(t,fundEquity)
            
        if value1=='Return' and value2 != 'fundEquity' :
            inx = markets.index(value2)-1
            subplt(t,exposure[:,inx],equity[:,inx])
        if value1=='Long' and value2 != 'fundEquity':
            inx = markets.index(value2)-1
            plot2(t[settings['lookback']-2:-1],returnLong[inx])
        if value1=='short' and value2 != 'fundEquity':
            inx = markets.index(value2)-1
            plot2(t[settings['lookback']-2:-1],returnShort[inx])
        if value1=='Markets' and value2 != 'fundEquity':
            inx = markets.index(value2)-1
            plot2(t[settings['lookback']-2:-1],returnShort[inx])

    def shutdown_interface():
        my_frame_1.pack_forget()
        my_frame_1.destroy()
        my_frame_2.pack_forget()
        my_frame_2.destroy()
        my_frame_3.pack_forget()
        my_frame_3.destroy()
#        quit=tk.messagebox.askyesno(title='Warning',message='Close the window?')
#        if quit==True:
#            PerformanceUI.destroy()
        PerformanceUI.eval('::ttk::CancelRepeat')
        PerformanceUI.quit()
        PerformanceUI.destroy()
    #--------------------------------------------------------------------------    
    PerformanceUI = tk.Tk()
    PerformanceUI.title('Trading System Performance')

    # creating a frame (my_frame_1)
    my_frame_1 = tk.Frame(PerformanceUI, relief=tk.SUNKEN)
    my_frame_1.pack(fill=tk.X,expand=1)
    
    my_label1  = tk.Label(my_frame_1, text="indicator")
    my_label1.pack(side=tk.LEFT,fill=tk.X,expand=0)   
    box_value1 = tk.StringVar()
    my_box1    = ttk.Combobox(my_frame_1, textvariable = box_value1, state = 'readonly')
    my_box1['values'] = ['Return','Equity','Markets','Long', 'Short']
    my_box1.pack(side=tk.LEFT,fill=tk.X,expand=1)   
    my_box1.current(0)
    my_box1.bind('<<ComboboxSelected>>',newselection1)
    
    my_label2  = tk.Label(my_frame_1, text="code")
    my_label2.pack(side=tk.LEFT,fill=tk.X,expand=0)        
    box_value2 = tk.StringVar()
    my_box2    = ttk.Combobox(my_frame_1, textvariable = box_value2, state = 'readonly')
    my_box2['values'] = markets
    my_box2.current(0)
    my_box2.pack(side=tk.LEFT,fill=tk.X,expand=1)  
    my_box2.bind('<<ComboboxSelected>>',newselection1)
    
    # creating a frame (my_frame_2)
    my_frame_2 = tk.Frame(PerformanceUI, relief=tk.SUNKEN)
    my_frame_2.pack(side=tk.LEFT,fill=tk.X,expand=1)
    #设置图形尺寸与质量
    f = plt.figure(figsize = (14,8), dpi=100)              
    style.use("ggplot")  
    canvas = FigureCanvasTkAgg(f, master=my_frame_2)
    canvas.get_tk_widget().pack(fill=tk.BOTH,expand=1)  
    
    #把matplotlib绘制图形的导航工具栏显示到tkinter窗口上
    toolbar =NavigationToolbar2TkAgg(canvas, my_frame_2)
    toolbar.update()
    canvas._tkcanvas.pack(fill=tk.BOTH, expand=1)
    
    # creating a frame (my_frame_3)
    my_frame_3  = tk.Frame(PerformanceUI, relief=tk.SUNKEN)
    my_frame_3.pack(side=tk.LEFT,fill=tk.BOTH,expand=1)
    
    my_listbox1 = tk.Listbox(my_frame_3)
#    my_listbox1.insert(0, "python2")
    my_listbox1.pack(fill=tk.BOTH, expand=1)
    

    #--------------------------------------------------------------------------
    
    #绘制图形
#    a = f.add_subplot(111)
#    t = arange(0.0,3,0.01)
#    s = sin(2*pi*t)
#    a.plot(t, s)
#    canvas.show()
    
    plot2(t,fundEquity,plotStat=True)
    
    statistics = stats(equity[:,1])
    statlist   = []
    [statlist.extend([k,v]) for k,v in statistics.items()]
    for line in statlist:
        if not isinstance(line, str):
            line = np.around(line, decimals=3)
        my_listbox1.insert(tk.END, line)
        
    #--------------------------------------------------------------------------
    PerformanceUI.protocol("WM_DELETE_WINDOW", shutdown_interface)
    PerformanceUI.mainloop()    
 
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
            maxTimeOffPeak = np.max(offPeak)
            topIx          = np.argmax(offPeak)
        else:
            maxTimeOffPeak = 0
            topIx          = np.zeros(0)

        if np.not_equal(np.size(topIx),0):
            mtopStart= ixStart[topIx]-2
            mtopEnd  = ixEnd[topIx]-1

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