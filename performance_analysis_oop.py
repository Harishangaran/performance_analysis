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
    riskFreeRate = Annual risk free rate in float. eg: 0.05 for 5%
    threshold = Annual investor return threshod for Omega ratio calculation
                Value in float. eg: 0.1 for 10%

# Default arguments
    benchmarkTicker = SP 500
    performancePeriod = 1080 trading days
    riskFreeRate = 0%
    threshold = 0%

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
import seaborn as sns
import statsmodels.api as sm
from statsmodels import regression
from scipy.stats import norm
import matplotlib.ticker as mtick


class performanceAnalysis:
    def __init__(self,assetTicker,benchmarkTicker='^GSPC',
                 performancePeriod='1080d',riskFreeRate=0,threshold=0):
        self.assetTicker = assetTicker
        self.benchmarkTicker = benchmarkTicker
        self.performancePeriod = performancePeriod
        self.rf = riskFreeRate
        self.threshold = threshold
        self.thresholdlist = np.linspace(0,1,50)
        self.callPrice()
        self.returnStats()
        self.returnMetrics()
        self.plotPerfomanceAnalysis()
    
    def callPrice(self):
        
        # Call stock and benchmark price
        self.assetDf = yf.Ticker(self.assetTicker).history(
                                    period=self.performancePeriod)
        self.benchmarkDf = yf.Ticker(self.benchmarkTicker).history(
                                        period=self.performancePeriod)
        return self.assetDf, self.benchmarkDf
        
    def returnStats(self):
        # Calculate daily and cummultative returns
        self.assetDf['Daily Return'] = self.assetDf['Close'].pct_change(1)
        self.assetDf['Cumm Return'] = (1 + self.assetDf['Daily Return']
                                       ).cumprod()
        self.assetDf['Plot Return'] = (1 + self.assetDf['Daily Return']
                                       ).cumprod(
                                           ) - 1 # minus to run curve from 0%
        
        self.benchmarkDf['Daily Return'] =self.benchmarkDf['Close'
                                                           ].pct_change(1)
        self.benchmarkDf['Cumm Return'] = (1 + self.benchmarkDf['Daily Return']
                                           ).cumprod()
        self.benchmarkDf['Plot Return'] = (1 + self.benchmarkDf['Daily Return']
                                           ).cumprod() -1
        
        # Drawdown
        self.assetDf['Total Return'] = self.assetDf['Daily Return'].cumsum()
        self.assetDf['Drawdown'] = self.assetDf['Total Return'] - \
                                        self.assetDf['Total Return'].cummax()
        
        self.benchmarkDf['Total Return'] = self.benchmarkDf['Daily Return'
                                                            ].cumsum()
        self.benchmarkDf['Drawdown'] = self.benchmarkDf['Total Return'] - \
                                        self.benchmarkDf['Total Return'
                                                         ].cummax()
        
        # Monthly return
        self.assetDfM = self.assetDf.resample('1M').sum()
        self.assetDfM['Month'] = self.assetDfM.index.strftime('%b')
        self.assetDfM['Year'] = self.assetDfM.index.year
        self.assetDfM.rename(columns={'Daily Return':'Monthly Return'}, 
                             inplace=True)
        
        # Pivot table for heatmap plot
        # Multiplied by 100 to get percentage values
        self.pivotTable = (self.assetDfM.pivot_table(index="Month",
                        columns="Year",values="Monthly Return").fillna(0))*100
        columnOrder = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        self.pivotTable = self.pivotTable.reindex(columnOrder)
        
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
        negativeVol = df[df['Daily Return']< 0]['Daily Return'].std(
                                                            ) * np.sqrt(252)
        sortino = (self.CAGR(df) - self.rf)/negativeVol
        return sortino
    
    def MaxDrawdown(self,df):
        maxdd = -(df['Drawdown'].min()) #should be postive
        return maxdd
    
    def Calmar(self,df):
        calmar = self.CAGR(df)/self.MaxDrawdown(df)
        return calmar

    def Omega(self,df,threshold):
        # Get daily threshold from annualised threshold value
        dailyThreshold = (threshold + 1) ** np.sqrt(1/252) - 1
        
        # Get excess return
        df['Excess'] = df['Daily Return'] - dailyThreshold
        
        # Get sum of all values excess return above 0
        dfPositiveSum = (df[df['Excess'] > 0].sum())['Excess']
        
        # Get sum of all values excess return below 0
        dfNegativeSum = (df[df['Excess'] < 0].sum())['Excess']
    
        omega = dfPositiveSum/(-dfNegativeSum)
        
        return omega
    
   
    def OmegaCurve(self,df):
        omegaValues = []
        
        for i in self.thresholdlist:
            val = round(self.Omega(df,i),10)
            omegaValues.append(val)
    
        return omegaValues
    
    def alpha_beta(self,benchmarkdf,assetdf):
        
        # Get asset and benchmark return
        asset_ret = assetdf['Close'].pct_change(1)[1:]
        bench_ret = benchmarkdf['Close'].pct_change(1)[1:]
        
        benchmark = sm.add_constant(bench_ret)
        
        model = regression.linear_model.OLS(asset_ret,benchmark).fit()
        
        self.rqsr = model.rsquared
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
        varDaily = -(norm.ppf(1-percent,mean,sd))
        return varDaily
           
    def returnMetrics(self):
        ''' Stats '''
        assetReturn = "{:.2%}".format((self.assetDf['Close'][-1] - \
                    self.assetDf['Close'][0])/self.assetDf['Close'][0])
        benchReturn = "{:.2%}".format((self.benchmarkDf['Close'][-1] - \
                self.benchmarkDf['Close'][0])/self.benchmarkDf['Close'][0])
        
        assetDailyMeanReturn = "{:.2%}".format(
                                        self.assetDf['Daily Return'].mean())
        benchDailyMeanReturn = "{:.2%}".format(
                                    self.benchmarkDf['Daily Return'].mean())
        
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
        
        assetOmega = round(self.Omega(self.assetDf,self.threshold),2)
        benchOmega = round(self.Omega(self.benchmarkDf,self.threshold),2)
        
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
        
        assetList = [assetReturn,assetDailyMeanReturn,assetCAGR,
                     assetVol,assetSharpe,assetSortino,assetMDD,
                     assetCalmar,assetOmega,assetTreynor,assetIR,
                     assetVar90,assetVar95,assetVar99,assetAlpha,assetBeta]
        
        benchList = [benchReturn,benchDailyMeanReturn,benchCAGR,
                     benchVol,benchSharpe,benchSortino,benchMDD,
                     benchCalmar,benchOmega,'-','-',benchVar90,benchVar95,
                     benchVar99,'-','-']
        
        self.data = pd.DataFrame()
        self.data['Metrics'] = ['Total Return','Daily Return Mean','CAGR',
                                'Annual Std','Sharpe Ratio','Sortino Ratio',
                                'Max Drawdown','Calmar Ratio','Omega Ratio',
                                'Treynor Ratio','Information Ratio',
                                'Daily VaR at 90%','Daily VaR at 95%',
                                'Daily VaR at 99%','Alpha','Beta']
        self.data[self.assetTicker] = assetList
        self.data['Benchmark'] = benchList
        
        return self.data
    
    def plotPerfomanceAnalysis(self):
        plt.style.use('seaborn')
        
        # Calculate number of months of the performance period - used in title
        month = int((self.assetDf.index[-1] - self.assetDf.index[0]
                                                 )/np.timedelta64(1,'M'))
        title = 'The Complete Performance Report of'\
            ' {} Stock Over the Last {} months'.format(self.assetTicker,month)
        
        fig,axes = plt.subplots(nrows=4,ncols=2,figsize=(30,25),dpi=300)
        fig.suptitle(title.upper(),fontweight='bold',fontsize=20)
        
        # Plot all scenarios
        # plot 1 cummulative graph
        axes[0,0].plot(self.assetDf.index,self.assetDf['Plot Return'],
                                               label=self.assetTicker)
        axes[0,0].plot(self.benchmarkDf.index,self.benchmarkDf['Plot Return']
                                                           ,label='Benchmark')
        axes[0,0].legend() #add legend
        axes[0,0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        axes[0,0].set_title('Cummulative Return',fontweight='bold')
        
        # Daily returns
        axes[0,1].plot(self.assetDf['Daily Return'][1:],label='Asset')
        axes[0,1].set_title('Daily Return of {}'.format(self.assetTicker),
                                                        fontweight='bold')
        #vals = axes[0,1].get_yticks()
        # Use PercentFormatter(1.0) multiply values by 100
        axes[0,1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        
        # Distribution of returns
        axes[2,0].hist(self.assetDf['Daily Return'][1:],bins=int(0.5*len(
                                                            self.assetDf)))
        self.assetDf['Daily Return'][1:].plot.kde(ax=axes[2,0])
        axes[2,0].xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        axes[2,0].set_title('Distribution of Daily Return of {}'.format(
            self.assetTicker),fontweight='bold')
        
        # Underwater plot of drawdown
        self.assetDf['Drawdown'].plot.area(ax=axes[3,0])
        axes[3,0].set_title('Underwater plot of drawdown of {}'.format(
            self.assetTicker),fontweight='bold')
        axes[3,0].tick_params(labelrotation=0)
        axes[3,0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        axes[3,0].set_xlabel('')
        
        # Regression
        axes[1,0].scatter(self.assetDf['Daily Return'],
                          self.benchmarkDf['Daily Return'],alpha=0.25)
        axes[1,0].plot(self.assetDf['Daily Return'],(
            self.alpha_beta(self.benchmarkDf,self.assetDf)[0] + (
                self.alpha_beta(self.benchmarkDf,self.assetDf
                                )[1]*self.assetDf['Daily Return'])),c='r')
        axes[1,0].set_title('Linear regression against benchmark',
                            fontweight='bold')
        axes[1,0].annotate('R-SQR = {}'.format(round(self.rqsr,2)), 
                         xy=(1, 0), xycoords='axes fraction', 
                         fontsize=8, horizontalalignment='left', 
                         xytext=(-0.1, 0.1),
                         textcoords='data',
                         verticalalignment='top')
        axes[1,0].set_ylabel(self.assetTicker,fontsize=10)
        axes[1,0].set_xlabel('Benchmark',fontsize=10)
        
        # Omega curve
        axes[1,1].plot(self.thresholdlist,self.OmegaCurve(
                                        self.assetDf),label=self.assetTicker)
        axes[1,1].plot(self.thresholdlist,self.OmegaCurve(
                                        self.benchmarkDf),label='Benchmark')
        axes[1,1].legend() #add legend
        axes[1,1].set_ylabel('Omega Ratio',fontsize=10)
        axes[1,1].set_xlabel('Annual Threshold',fontsize=10)
        axes[1,1].xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        axes[1,1].set_title('Omega Curve',fontweight='bold')
        
        # Monthly return heatmap
        sns.heatmap(self.pivotTable, ax=axes[2,1],annot=True,
                    cmap="GnBu",cbar_kws={'format': '%.0f%%'})
        axes[2,1].set_title('Heatmap of {} Monthly Return'.format(
                                self.assetTicker),fontweight='bold')
        
        axes[3,1].axis('tight')
        axes[3,1].axis('off')
        table = axes[3,1].table(cellText=self.data.values,colLabels=self.data.columns,
        cellLoc='center',loc='center',
        colColours=['lightsteelblue','lightsteelblue','lightsteelblue'])
        axes[3,1].set_title('Performance Stats',fontweight='bold')
        
        for key, cell in table.get_celld().items():
            cell.set_edgecolor('w')
            cell.set_facecolor('#f2f2fc')
            cell.set_linewidth(2)
        table.get_celld()[(0,0)].set_facecolor('lavender')
        table.get_celld()[(0,1)].set_facecolor('lavender')
        table.get_celld()[(0,2)].set_facecolor('lavender')
        # Adjust distance of subplots from title
        fig.subplots_adjust(top=0.95)
        


performanceAnalysis('MSFT','^GSPC','1080d',0,0)
