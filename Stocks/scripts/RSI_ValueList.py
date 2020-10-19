import pandas as pd
import math
from talib import RSI, BBANDS, MACD
import datetime
import sys


stockListPath = 'D:\\Personal\\share\\BasicData\\ALL-stocks.txt'
dataFolderPath = 'D:\\Personal\\share\\data'
resultFolderPath = 'D:\\Personal\\share\\results'

def readStockList(path):
    dataFile = open(path, "r")
    dataLines = dataFile.readlines()
    dataLines = [line.rstrip() for line in dataLines]
    dataFile.close()
    return dataLines

def CalculateIndices(df):
    MACD_FAST = 10
    MACD_SLOW = 21
    MACD_SIGNAL = 7

    macd, macdSignal, macdHist = MACD(df['c'], MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    rsi10 = RSI(df['c'], 10)
    rsi14 = RSI(df['c'], 14)

    return macd, macdSignal, macdHist, rsi10, rsi14

def RunForStock(stockName):
    dataFileName = dataFolderPath + '\\'+ stockName + '.txt'
    df = pd.read_json(dataFileName)
    df['t'] = pd.to_datetime(df['t'], unit='s')
    df['t'] = df['t'].dt.date

    if len(df.index) == 0:
        return [stockName, 0, 0, 0, 0, 0]

    
    macd, macdSignal, macdHist, rsi10, rsi14 = CalculateIndices(df)
    return [stockName, macd[len(macd)-1], macdSignal[len(macdSignal)-1], macdHist[len(macdHist)-1], rsi10[len(rsi10)-1], rsi14[len(rsi14)-1]]

stockList = readStockList(stockListPath)
signalData = []

for stockName in stockList:
    tmpDf = RunForStock(stockName)
    #print(tmpDf)
    signalData.append(tmpDf)

columns = ['Stock', 'macd', 'macdSignal', 'macdHist', 'rsi10', 'rsi14']
data = pd.DataFrame(signalData, columns=columns)
data.to_csv(resultFolderPath + '\\BS_MACD_RSI_Data.csv')
print(signalData)