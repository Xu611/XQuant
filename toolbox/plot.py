# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 10:38:09 2018

@author: Administrator

tutorial:
        command: to call a function
        For PyQt5:pyuic5 -x example.ui -o example.py
"""              
import numpy as np
import datetime
import os 

from dataToolkit import stats

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import style
import matplotlib.pyplot as plt

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
def plotts(tradingSystem, equity,mEquity,exposure,settings,DATE,statistics,returns,marketReturns):
    ''' plots equity curve and calculates trading system statistics
        画出权益曲线
    Args:
        tradingSystem(str): .py file name
        equity (list): list of equity of evaluated trading system.
        mEquity (list): list of equity of each market over the trading days.
        exposure (list): list of positions over the trading days.
        settings (dict): list of settings.
        DATE (list): list of dates corresponding to entries in equity.
        statistics(dict): Statistics for Analyzing Trading System
    Copyright Quantiacs LLC - March 2015
    '''
    # Initialize selected index of the two dropdown lists
    style.use("ggplot")                                                 # emulates the aesthetics of ggplot 
    global indx_TradingPerf, indx_Exposure, indx_MarketRet 
    cashOffset = 0
    inx  = [0]
    inx2 = [0]
    inx3 = [0]
    indx_TradingPerf = 0
    indx_Exposure = 0
    indx_MarketRet = 0

    mRetMarkets = list(settings['markets'])

    try:
        mRetMarkets.remove('CASH')
        cashOffset = 1
    except:
        pass

    settings['markets'].insert(0,'fundEquity')                          # ['fundEquity',...]

    DATEord=[]
    lng = len(DATE)
    for i in range(lng):
        #DATEord.append(datetime.datetime.strptime(str(DATE[i]),'%Y%m%d')) # 旧代码 
        DATEord.append(datetime.datetime.strptime(DATE[i],'%Y-%m-%d'))  # 由于DATE数据修改为了字符串格式的列表，所以对此段代码进行修改

    # Prepare all the y-axes
    equityList = np.transpose(np.array(mEquity))

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



    def plot(indx_TradingPerf,indx_Exposure):
        plt.clf()

        Subplot_Equity = plt.subplot2grid((8,8), (0,0), colspan = 6, rowspan = 6)
        Subplot_Exposure = plt.subplot2grid((8,8), (6,0), colspan = 6, rowspan = 2, sharex = Subplot_Equity)
        t = np.array(DATEord)

        if indx_TradingPerf == 0:   # fundEquity selected
            lon = Long.sum(axis=0)
            sho = Short.sum(axis=0)
            y_Long = lon
            y_Short = sho
            if indx_Exposure == 0:  # Long & Short selected
                y_Equity = equity
                Subplot_Equity.plot(t,y_Equity,'b',linewidth=0.5)
            elif indx_Exposure == 1:  # Long Selected
                y_Equity = settings['budget'] * np.cumprod(1+returnLong.sum(axis = 0))
                Subplot_Equity.plot(t,y_Equity,'c',linewidth=0.5)
            else:      # Short Selected
                y_Equity = settings['budget'] * np.cumprod(1+returnShort.sum(axis = 0))
                Subplot_Equity.plot(t,y_Equity,'g',linewidth=0.5)
            statistics=stats(y_Equity)
            Subplot_Equity.plot(DATEord[statistics['maxDDBegin']:statistics['maxDDEnd']+1],y_Equity[statistics['maxDDBegin']:statistics['maxDDEnd']+1],color='red',linewidth=0.5, label = 'Max Drawdown')
            # Subplot_Equity.plot(DATEord[(statistics['maxTimeOffPeakBegin']+1):(statistics['maxTimeOffPeakBegin']+statistics['maxTimeOffPeak']+2)],y_Equity[statistics['maxTimeOffPeakBegin']+1]*np.ones((statistics['maxTimeOffPeak']+1)),'r--',linewidth=2, label = 'Max Time Off Peak')
            if not(np.isnan(statistics['maxTimeOffPeakBegin'])) and not(np.isnan(statistics['maxTimeOffPeak'])):
                Subplot_Equity.plot(DATEord[(statistics['maxTimeOffPeakBegin']+1):(statistics['maxTimeOffPeakBegin']+statistics['maxTimeOffPeak']+2)],y_Equity[statistics['maxTimeOffPeakBegin']+1]*np.ones((statistics['maxTimeOffPeak']+1)),'r--',linewidth=2, label = 'Max Time Off Peak')
            Subplot_Exposure.plot(t,y_Long,'c',linewidth=0.5, label = 'Long')
            Subplot_Exposure.plot(t,y_Short,'g',linewidth=0.5, label = 'Short')
            # Hide the Long(Short) curve in market exposure subplot when Short(Long) is plotted in Equity Curve subplot
            if indx_Exposure == 1:
                Subplot_Exposure.lines.pop(1)
            elif indx_Exposure == 2:
                Subplot_Exposure.lines.pop(0)
            Subplot_Equity.set_yscale('log')
            Subplot_Equity.set_ylabel('Performance (Logarithmic)')
        else:   # individual market selected
            y_Long = Long[indx_TradingPerf-1]
            y_Short = Short[indx_TradingPerf-1]
            if indx_Exposure == 0:            # Long & Short Selected
                y_Equity = equityList[indx_TradingPerf-1]
                Subplot_Equity.plot(t,y_Equity,'b',linewidth=0.5)
            elif indx_Exposure == 1:        # Long Selected
                y_Equity = np.cumprod(1+returnLong[indx_TradingPerf-1])
                Subplot_Equity.plot(t,y_Equity,'c',linewidth=0.5)
            else:                   # Short Selected
                y_Equity = np.cumprod(1+returnShort[indx_TradingPerf-1])
                Subplot_Equity.plot(t,y_Equity,'g',linewidth=0.5)
            statistics=stats(y_Equity)
            Subplot_Exposure.plot(t,y_Long,'c',linewidth=0.5, label = 'Long')
            Subplot_Exposure.plot(t,y_Short,'g',linewidth=0.5, label = 'Short')
            if indx_Exposure == 1:
                Subplot_Exposure.lines.pop(1)
            elif indx_Exposure == 2:
                Subplot_Exposure.lines.pop(0)
            if np.isnan(statistics['maxDDBegin']) == False:
                Subplot_Equity.plot(DATEord[statistics['maxDDBegin']:statistics['maxDDEnd']+1],y_Equity[statistics['maxDDBegin']:statistics['maxDDEnd']+1],color='red',linewidth=0.5, label = 'Max Drawdown')
                if not(np.isnan(statistics['maxTimeOffPeakBegin'])) and not(np.isnan(statistics['maxTimeOffPeak'])):
                    Subplot_Equity.plot(DATEord[(statistics['maxTimeOffPeakBegin']+1):(statistics['maxTimeOffPeakBegin']+statistics['maxTimeOffPeak']+2)],y_Equity[statistics['maxTimeOffPeakBegin']+1]*np.ones((statistics['maxTimeOffPeak']+1)),'r--',linewidth=2, label = 'Max Time Off Peak')
                    # Subplot_Equity.plot(DATEord[(statistics['maxTimeOffPeakBegin']+1):(statistics['maxTimeOffPeakBegin']+statistics['maxTimeOffPeak']+2)],y_Equity[statistics['maxTimeOffPeakBegin']+1]*np.ones((statistics['maxTimeOffPeak']+1)),'r--',linewidth=2, label = 'Max Time Off Peak')
            Subplot_Equity.set_ylabel('Performance')

        statsStr="Sharpe Ratio = {sharpe:.4f}\nSortino Ratio = {sortino:.4f}\n\nPerformance (%/yr) = {returnYearly:.4f}\nVolatility (%/yr)       = {volaYearly:.4f}\n\nMax Drawdown = {maxDD:.4f}\nMAR Ratio         = {mar:.4f}\n\nMax Time off peak =  {maxTimeOffPeak}\n\n\n\n\n\n".format(**statistics)


        Subplot_Equity.autoscale(tight=True)
        Subplot_Exposure.autoscale(tight=True)
        Subplot_Equity.set_title('Trading Performance of %s' %settings['markets'][indx_TradingPerf])
        Subplot_Equity.get_xaxis().set_visible(False)
        Subplot_Exposure.set_ylabel('Long/Short')
        Subplot_Exposure.set_xlabel('Year')
        Subplot_Equity.legend(bbox_to_anchor=(1.03, 0), loc='lower left', borderaxespad=0.)
        Subplot_Exposure.legend(bbox_to_anchor=(1.03, 0.63), loc='lower left', borderaxespad=0.)

        # Performance Numbers Textbox
        f.text(.72,.58,statsStr)

        plt.gcf().canvas.draw()


    def plot2(indx_Exposure, indx_MarketRet):
        plt.clf()

        MarketReturns = plt.subplot2grid((8,8), (0,0), colspan = 6, rowspan = 8)
        t = np.array(DATEord)

        if indx_Exposure == 2:
            mRet = np.cumprod(1-marketRet[indx_MarketRet + cashOffset])
        else:
            mRet = np.cumprod(1+marketRet[indx_MarketRet + cashOffset])

        MarketReturns.plot(t,mRet,'b',linewidth=0.5)
        statistics=stats(mRet)
        MarketReturns.set_ylabel('Market Returns')

        statsStr="Sharpe Ratio = {sharpe:.4f}\nSortino Ratio = {sortino:.4f}\n\nPerformance (%/yr) = {returnYearly:.4f}\nVolatility (%/yr)       = {volaYearly:.4f}\n\nMax Drawdown = {maxDD:.4f}\nMAR Ratio         = {mar:.4f}\n\n Max Time off peak =  {maxTimeOffPeak}\n\n\n\n\n\n".format(**statistics)

        MarketReturns.autoscale(tight=True)
        MarketReturns.set_title('Market Returns of %s' %mRetMarkets[indx_MarketRet])
        MarketReturns.set_xlabel('Date')

        # Performance Numbers Textbox
        f.text(.72,.58,statsStr)

        plt.gcf().canvas.draw()



    # 下拉列表的回调函数: Callback function for two dropdown lists
    def newselection(event):
        global indx_TradingPerf, indx_Exposure, indx_MarketRet
        value_of_combo = dropdown.current()
        inx.append(value_of_combo)
        indx_TradingPerf = inx[-1]
        indx_MarketRet = -1

        plot(indx_TradingPerf,indx_Exposure)

    def newselection2(event):
        global indx_TradingPerf, indx_Exposure, indx_MarketRet
        value_of_combo2 = dropdown2.current()
        inx2.append(value_of_combo2)
        indx_Exposure = inx2[-1]

        if indx_TradingPerf == -1:
            plot2(indx_Exposure,indx_MarketRet)
        else:
            plot(indx_TradingPerf,indx_Exposure)

    def newselection3(event):
        global indx_TradingPerf, indx_Exposure, indx_MarketRet
        value_of_combo3 = dropdown3.current()
        inx3.append(value_of_combo3)
        indx_MarketRet = inx3[-1]
        indx_TradingPerf = -1

        plot2(indx_Exposure, indx_MarketRet)

    def submit_callback():
        tsfolder, tsName = os.path.split(tradingSystem)
        submit(tradingSystem,tsName[:-3])


    def shutdown_interface():
        TradingUI.eval('::ttk::CancelRepeat')
        # TradingUI.destroy()
        TradingUI.quit()
        TradingUI.destroy()
        # sys.exit()

    # GUI mainloop
    TradingUI = tk.Tk()
    TradingUI.title('Trading System Performance')

    Label_1 = tk.Label(TradingUI, text="Trading Performance:")
    Label_1.grid(row = 0, column = 0, sticky = tk.EW)

    box_value = tk.StringVar()
    dropdown = ttk.Combobox(TradingUI, textvariable  = box_value, state = 'readonly')
    dropdown['values'] = settings['markets']
    dropdown.grid(row=0, column=1,sticky=tk.EW)
    dropdown.current(0)
    dropdown.bind('<<ComboboxSelected>>',newselection)

    Label_2 = tk.Label(TradingUI, text="Exposure:")
    Label_2.grid(row = 0, column = 2, sticky = tk.EW)

    box_value2 = tk.StringVar()
    dropdown2 = ttk.Combobox(TradingUI, textvariable  = box_value2, state = 'readonly')
    dropdown2['values'] = ['Long & Short', 'Long', 'Short']
    dropdown2.grid(row=0, column=3,sticky=tk.EW)
    dropdown2.current(0)
    dropdown2.bind('<<ComboboxSelected>>',newselection2)

    Label_3 = tk.Label(TradingUI, text="Market Returns:")
    Label_3.grid(row = 0, column = 4, sticky = tk.EW)

    box_value3 = tk.StringVar()
    dropdown3 = ttk.Combobox(TradingUI, textvariable  = box_value3, state = 'readonly')
    dropdown3['values'] = mRetMarkets
    dropdown3.grid(row=0, column=5,sticky=tk.EW)
    dropdown3.current(0)
    dropdown3.bind('<<ComboboxSelected>>',newselection3)

    f = plt.figure(figsize = (14,8))
    canvas = FigureCanvasTkAgg(f, master=TradingUI)



    if False:  # updateCheck():
        pass
    else:      # Canvas它是用于绘制的图片和其他复杂的布局图形、文本和控件
        canvas.get_tk_widget().grid(row=1,column=0,columnspan = 6, rowspan = 2, sticky=tk.NSEW)


    button_submit = tk.Button(TradingUI, text = 'Submit Trading System', command = submit_callback)
    button_submit.grid(row = 4, column = 0, columnspan = 6, sticky = tk.EW)

    Subplot_Equity = plt.subplot2grid((8,8), (0,0), colspan = 6, rowspan = 6)
    Subplot_Exposure = plt.subplot2grid((8,8), (6,0), colspan = 6, rowspan = 2, sharex = Subplot_Equity)
    t = np.array(DATEord)

    lon = Long.sum(axis=0)
    sho = Short.sum(axis=0)
    y_Long = lon
    y_Short = sho
    y_Equity = equity
    Subplot_Equity.plot(t,y_Equity,'b',linewidth=0.5)
    statistics=stats(y_Equity)
    Subplot_Equity.plot(DATEord[statistics['maxDDBegin']:statistics['maxDDEnd']+1],y_Equity[statistics['maxDDBegin']:statistics['maxDDEnd']+1],color='red',linewidth=0.5, label = 'Max Drawdown')
    # Subplot_Equity.plot(DATEord[(statistics['maxTimeOffPeakBegin']+1):(statistics['maxTimeOffPeakBegin']+statistics['maxTimeOffPeak']+2)],y_Equity[statistics['maxTimeOffPeakBegin']+1]*np.ones((statistics['maxTimeOffPeak']+1)),'r--',linewidth=2, label = 'Max Time Off Peak')
    if not(np.isnan(statistics['maxTimeOffPeakBegin'])) and not(np.isnan(statistics['maxTimeOffPeak'])):
        Subplot_Equity.plot(DATEord[(statistics['maxTimeOffPeakBegin']+1):(statistics['maxTimeOffPeakBegin']+statistics['maxTimeOffPeak']+2)],y_Equity[statistics['maxTimeOffPeakBegin']+1]*np.ones((statistics['maxTimeOffPeak']+1)),'r--',linewidth=2, label = 'Max Time Off Peak')
    Subplot_Exposure.plot(t,y_Long,'c',linewidth=0.5, label = 'Long')
    Subplot_Exposure.plot(t,y_Short,'g',linewidth=0.5, label = 'Short')
    Subplot_Equity.set_yscale('log')
    Subplot_Equity.set_ylabel('Performance (Logarithmic)')

    statsStr="Sharpe Ratio = {sharpe:.4f}\nSortino Ratio = {sortino:.4f}\n\nPerformance (%/yr) = {returnYearly:.4f}\nVolatility (%/yr)       = {volaYearly:.4f}\n\nMax Drawdown = {maxDD:.4f}\nMAR Ratio         = {mar:.4f}\n\n Max Time off peak =  {maxTimeOffPeak}\n\n\n\n\n\n".format(**statistics)

    Subplot_Equity.autoscale(tight=True)
    Subplot_Exposure.autoscale(tight=True)
    Subplot_Equity.set_title('Trading Performance of %s' %settings['markets'][indx_TradingPerf])
    Subplot_Equity.get_xaxis().set_visible(False)
    Subplot_Exposure.set_ylabel('Long/Short')
    Subplot_Exposure.set_xlabel('Year')
    Subplot_Equity.legend(bbox_to_anchor=(1.03, 0), loc='lower left', borderaxespad=0.)
    Subplot_Exposure.legend(bbox_to_anchor=(1.03, 0.63), loc='lower left', borderaxespad=0.)

    plt.gcf().canvas.draw()
    f.text(.72,.58,statsStr)


    TradingUI.protocol("WM_DELETE_WINDOW", shutdown_interface)

    TradingUI.mainloop()    # 运行Tk程序

#%%
def submit():
    '''  提交系统到服务器...
    '''
    pass
