import pandas as pd
import math
from talib import RSI, BBANDS, MACD
import datetime
import sys
from Templating.StocksCommon import CalcBuyCost, CalcSellReturn, BuyQtyForPrice
from Templating.PlotChart import PlotStockChart


stockListPath = 'D:\\Personal\\share\\BasicData\\ALL-stocks.txt'
dataFolderPath = 'D:\\Personal\\share\\data'
resultFolderPath = 'D:\\Personal\\share\\results'

START_TRADING_ON = datetime.date(2020, 5, 14)

NO_OF_SHARES_KEY = 'NoOfShares'
CASH_KEY = 'Cash'
CASH_VALUE = 100000
ind_columns = ['MACD', 'MACDS', 'MACDH']

def CalculateIndices(df):
    MACD_FAST = 10
    MACD_SLOW = 21
    MACD_SIGNAL = 7

    macd, macdSignal, macdHist = MACD(df['c'], MACD_FAST, MACD_SLOW, MACD_SIGNAL)

    df[ind_columns[0]] = macd
    df[ind_columns[1]] = macdSignal
    df[ind_columns[2]] = macdHist
    return df

def TestForBuySignal(index, dataSet):
    if math.isnan(dataSet.at[index, ind_columns[2]]):
        return False

    if dataSet.at[index, ind_columns[2]] <= 0:
        if dataSet.at[index, ind_columns[2]] > dataSet.at[index-1, ind_columns[2]] :
            return True
    return False

def TestForSellSignal(index, dataSet):
    if  dataSet.at[index, ind_columns[2]] <= 0:
        if dataSet.at[index -1, ind_columns[2]] > 0:
            return True

        if  dataSet.at[index -1, ind_columns[2]] < 0 and dataSet.at[index -1, ind_columns[2]] > dataSet.at[index, ind_columns[2]]:
            return True
    return False

def readStockList(path):
    dataFile = open(path, "r")
    dataLines = dataFile.readlines()
    dataLines = [line.rstrip() for line in dataLines]
    dataFile.close()
    return dataLines

def RunForStock(stockName):
    dataFileName = dataFolderPath + '\\'+ stockName + '.txt'
    df = pd.read_json(dataFileName)
    df['t'] = pd.to_datetime(df['t'], unit='s')
    df['t'] = df['t'].dt.date

    if len(df.index) == 0:
        return [stockName, '1900-01-01', 'Idle']

    dataSet = CalculateIndices(df)
    index = dataSet.last_valid_index()

    #for index, row in dataSet.iterrows():
    #if index > 0 and row['t'] >= START_TRADING_ON:
    if TestForBuySignal(index, dataSet):
        return [stockName, df.iloc[index]['t'], 'Buy', df.iloc[index]['MACDH'],df.iloc[index]['MACD'],df.iloc[index]['MACDS']]
    elif TestForSellSignal(index, dataSet):
        return [stockName, df.iloc[index]['t'], 'Sell', df.iloc[index]['MACDH'],df.iloc[index]['MACD'],df.iloc[index]['MACDS']]
    return [stockName, df.iloc[index]['t'], 'Idle', df.iloc[index]['MACDH'],df.iloc[index]['MACD'],df.iloc[index]['MACDS']]

stockList = readStockList(stockListPath)
signalData = []

for stockName in stockList:
    tmpDf = RunForStock(stockName)
    signalData.append(tmpDf)

columns = ['Stock','Date','Signal','MACDH','MACD','MACDS']
data = pd.DataFrame(signalData, columns=columns)
data.to_csv(resultFolderPath + '\\BS_MACD_Only.csv')

for index, row in data.iterrows():
    if row.Signal == 'Buy' :
        PlotStockChart(row.Stock)
print('Done ')