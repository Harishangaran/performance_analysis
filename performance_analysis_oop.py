# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 17:30:17 2020

@author: harishangaran

Stock Performance Analysis

The following script calls stock and benchmark indice price from yahoo finance
to create a complete performance report.

Calling the class:

# Positional Arguments
    assetTicker = Ticker symbol for the stock
    benchmarkTicker = Ticker symbol for benchmark
    performancePeriod = The time period in days of the investment period
    riskFreeRate = Annual risk free rate eg. 0.05 for 5%
    threshold = Annual investor return threshod for Omega ratio calculation
                Value is in percent

# Call eg:
    performanceAnalysis(stockTicker,benchmarkTicker,performancePeriod,
    riskFreeRate,threshold)          
    performanceAnalysis('MSFT','^GSPC','1000d',0.0025,0)

    # Outputs
        matplotlib plot performance summary

"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import statsmodels.api as sm

from statsmodels import regression
from scipy.stats import norm


class performanceAnalysis:
    def __init__(self,assetTicker,benchmarkTicker,performancePeriod,riskFreeRate,threshold):
        self.assetTicker = assetTicker
        self.benchmarkTicker = benchmarkTicker
        self.performancePeriod = performancePeriod
        self.rf = riskFreeRate
        self.threshold = threshold
        self.callPrice()
        self.returnStats()
        self.returnMetrics()
        self.plotPerfomanceAnalysis()
    
    def callPrice(self):
        
        # Call stock and benchmark price
        self.assetDf = yf.Ticker(self.assetTicker).history(period=self.performancePeriod)
        self.benchmarkDf = yf.Ticker(self.benchmarkTicker).history(period=self.performancePeriod)
        return self.assetDf, self.benchmarkDf
        
    def returnStats(self):
        # Calculate daily and cummultative returns
        self.assetDf['Daily Return'] = self.assetDf['Close'].pct_change(1)
        self.assetDf['Cumm Return'] = (1 + self.assetDf['Daily Return']).cumprod()

        self.benchmarkDf['Daily Return'] =self.benchmarkDf['Close'].pct_change(1)
        self.benchmarkDf['Cumm Return'] = (1 + self.benchmarkDf['Daily Return']).cumprod()

        ''' Drawdown '''
        self.assetDf['Total Return'] = self.assetDf['Daily Return'].cumsum()
        self.assetDf['Drawdown'] = self.assetDf['Total Return'] - self.assetDf['Total Return'].cummax()
        
        self.benchmarkDf['Total Return'] = self.benchmarkDf['Daily Return'].cumsum()
        self.benchmarkDf['Drawdown'] = self.benchmarkDf['Total Return'] - self.benchmarkDf['Total Return'].cummax()
        
        return self.assetDf, self.benchmarkDf

    def CAGR(self,df):
        lenOfPeriod = len(df)/252
        CAGR = (df['Cumm Return'][-1]) ** (1/lenOfPeriod) - 1
        return CAGR
    
    def Volatility(self,df):
        vol = df['Daily Return'].std() * np.sqrt(252)
        return vol

    def Sharpe(self,df):
        sharpe = (self.CAGR(df)-self.rf)/self.Volatility(df)
        return sharpe

    def Sortino(self,df):
        negativeVol = df[df['Daily Return']< 0]['Daily Return'].std() * np.sqrt(252)
        sortino = (self.CAGR(df) - self.rf)/negativeVol
        return sortino
    
    def MaxDrawdown(self,df):
        maxdd = -(df['Drawdown'].min()) #should be postive
        return maxdd
    
    def Calmar(self,df):
        calmar = self.CAGR(df)/self.MaxDrawdown(df)
        return calmar

    def Omega(self,df):
        # Get daily threshold from annualised threshold value
        dailyThreshold = (self.threshold + 1) ** np.sqrt(1/252) - 1
        
        # Get excess return
        df['Excess'] = df['Daily Return'] - dailyThreshold
        
        # Get sum of all values excess return above 0
        dfPositiveSum = (df[df['Excess'] > 0].sum())['Excess']
        
        # Get sum of all values excess return below 0
        dfNegativeSum = (df[df['Excess'] < 0].sum())['Excess']
    
        omega = dfPositiveSum/(-dfNegativeSum)
        
        return omega
    
    def alpha_beta(self,benchmarkdf,assetdf):
        
        # Get asset and benchmark return
        asset_ret = assetdf['Close'].pct_change(1)[1:]
        bench_ret = benchmarkdf['Close'].pct_change(1)[1:]
        
        benchmark = sm.add_constant(bench_ret)
        
        model = regression.linear_model.OLS(asset_ret,benchmark).fit()
        
        # Return alpha and beta from regression
        return model.params[0],model.params[1]
    
    #Treynor ratio
    def Treynor(self,assetdf,benchdf):
        alpha,beta = self.alpha_beta(benchdf,assetdf)
        treynor = (self.CAGR(assetdf)- self.rf)/beta
        return treynor
    
    #Information ratio
    def InformationRatio(self,assetdf,benchdf):
        information = abs(self.CAGR(assetdf) - self.CAGR(benchdf
                        ))/abs(self.Volatility(assetdf) - self.Volatility(benchdf))   
        return information
    
    #VaR - percentiles 90,95,99
    def VaR(self,df,percentile):
        percent = percentile/100
        mean = np.mean(df['Daily Return'])
        sd = np.std(df['Daily Return'])
        varDaily = norm.ppf(1-percent,mean,sd)
        return varDaily

    def returnMetrics(self):
        ''' Stats '''
        assetReturn = "{:.2%}".format(self.assetDf['Daily Return'].sum())
        benchReturn = "{:.2%}".format(self.benchmarkDf['Daily Return'].sum())
        
        assetDailyMeanReturn = "{:.2%}".format(self.assetDf['Daily Return'].mean())
        benchDailyMeanReturn = "{:.2%}".format(self.benchmarkDf['Daily Return'].mean())
        
        assetCAGR = "{:.2%}".format(self.CAGR(self.assetDf))
        benchCAGR = "{:.2%}".format(self.CAGR(self.benchmarkDf))
        
        assetVol = "{:.2%}".format(self.Volatility(self.assetDf))
        benchVol = "{:.2%}".format(self.Volatility(self.benchmarkDf))
        
        assetSharpe = round(self.Sharpe(self.assetDf),2)
        benchSharpe = round(self.Sharpe(self.benchmarkDf),2)
        
        assetSortino = round(self.Sortino(self.assetDf),2)
        benchSortino = round(self.Sortino(self.benchmarkDf),2)
        
        assetMDD = "{:.2%}".format(self.MaxDrawdown(self.assetDf))
        benchMDD = "{:.2%}".format(self.MaxDrawdown(self.benchmarkDf))
        
        assetCalmar = round(self.Calmar(self.assetDf),2)
        benchCalmar = round(self.Calmar(self.benchmarkDf),2)
        
        assetOmega = round(self.Omega(self.assetDf),2)
        benchOmega = round(self.Omega(self.benchmarkDf),2)
        
        assetTreynor = round(self.Treynor(self.assetDf,self.benchmarkDf),2)
        
        assetIR = round(self.InformationRatio(self.assetDf,self.benchmarkDf),2)
        
        assetVar90 = "{:.2%}".format(self.VaR(self.assetDf,90))
        benchVar90 = "{:.2%}".format(self.VaR(self.benchmarkDf,90))
        
        assetVar95 = "{:.2%}".format(self.VaR(self.assetDf,95))
        benchVar95 = "{:.2%}".format(self.VaR(self.benchmarkDf,95))
        
        assetVar99 = "{:.2%}".format(self.VaR(self.assetDf,99))
        benchVar99 = "{:.2%}".format(self.VaR(self.benchmarkDf,99))
        
        assetAlpha = round(self.alpha_beta(self.benchmarkDf,self.assetDf)[0],3)
        
        assetBeta = round(self.alpha_beta(self.benchmarkDf,self.assetDf)[1],2)
        
        assetList = [assetReturn,assetDailyMeanReturn,assetCAGR,assetVol,assetSharpe,assetSortino,assetMDD,assetCalmar,assetOmega,assetTreynor,assetIR,assetVar90,assetVar95,assetVar99,assetAlpha,assetBeta]
        
        benchList = [benchReturn,benchDailyMeanReturn,benchCAGR,benchVol,benchSharpe,benchSortino,benchMDD,benchCalmar,benchOmega,'-','-',benchVar90,benchVar95,benchVar99,'-','-']
        
        self.data = pd.DataFrame()
        self.data['Metrics'] = ['Total Return','Daily Return Mean','CAGR','Annual Std','Sharpe Ratio','Sortino Ratio','Max Drawdown','Calmar Ratio','Omega Ratio','Treynor Ratio','Information Ratio','Daily VaR at 90%','Daily VaR at 95%','Daily VaR at 99%','Alpha','Beta']
        self.data[self.assetTicker] = assetList
        self.data['Benchmark'] = benchList
        
        return self.data
    
    def plotPerfomanceAnalysis(self):
        plt.style.use('seaborn')
        
        # Calculate number of months of the performance period - used in title
        month = int((self.assetDf.index[-1] - self.assetDf.index[0])/np.timedelta64(1,'M'))
        
        fig,axes = plt.subplots(nrows=5,ncols=1,figsize=(16,25),dpi=300)
        fig.suptitle('The Complete Performance Report of {} Stock Over Last {} months'.format(self.assetTicker,month))
        
        # Plot all scenarios
        # plot 1 cummulative graph
        axes[0].plot(self.assetDf.index,self.assetDf['Cumm Return'],label=self.assetTicker)
        axes[0].plot(self.benchmarkDf.index,self.benchmarkDf['Cumm Return'],label='Benchmark')
        axes[0].legend() #add legend
        axes[0].set_title('Cummulative Return')
        
        # Daily returns
        axes[1].plot(self.assetDf['Daily Return'][1:],label='Asset')
        axes[1].set_title('Daily Return')
        
        # Distribution of returns
        axes[2].hist(self.assetDf['Daily Return'][1:],bins=int(0.5*len(self.assetDf)))
        self.assetDf['Daily Return'][1:].plot.kde(ax=axes[2])
        axes[2].set_title('Distribution of Return')
        
        # Underwater plot of drawdown
        self.assetDf['Drawdown'].plot.area(ax=axes[3])
        axes[3].set_title('Underwater plot of drawdown')
        axes[3].tick_params(labelrotation=0)
        axes[3].set_xlabel('')
        
        axes[4].axis('tight')
        axes[4].axis('off')
        axes[4].table(cellText=self.data.values,colLabels=self.data.columns,cellLoc='center',loc='center',colColours=['lightsteelblue','lightsteelblue','lightsteelblue'])
        axes[4].set_title('Performance Stats')
        # Adjust distance of subplots from title
        fig.subplots_adjust(top=0.95)


performanceAnalysis('MSFT','^GSPC','1000d',0,0)