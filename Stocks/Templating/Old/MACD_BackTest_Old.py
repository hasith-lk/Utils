import pandas as pd
import math
from talib import RSI, BBANDS, MACD
import datetime
from Templating.StocksCommon import CalcBuyCost, CalcSellReturn, BuyQtyForPrice


stockListPath = 'D:\\Personal\\share\\BasicData\\ALL-stocks.txt'
dataFolderPath = 'D:\\Personal\\share\\data'
resultFolderPath = 'D:\\Personal\\share\\results'

START_TRADING_ON = datetime.date(2020, 1, 1)

NO_OF_SHARES_KEY = 'NoOfShares'
CASH_KEY = 'Cash'
CASH_VALUE = 100000
ind_columns = ['MACD', 'MACDS', 'MACDH']

def CalculateIndices(df):
    MACD_FAST = 10
    MACD_SLOW = 21
    MACD_SIGNAL = 5

    macd, macdSignal, macdHist = MACD(df['c'], MACD_FAST, MACD_SLOW, MACD_SIGNAL)

    df[ind_columns[0]] = macd
    df[ind_columns[1]] = macdSignal
    df[ind_columns[2]] = macdHist
    return df

def Exec_Buy(index, dataSet, stockAccount):
    if math.isnan(dataSet.loc[index, ind_columns[2]]):
        return stockAccount

    if dataSet.loc[index, ind_columns[2]] <= 0:
        if dataSet.loc[index, ind_columns[2]] > dataSet.loc[index-1, ind_columns[2]] :
            nShares, costDiff  = BuyQtyForPrice(dataSet.loc[index, 'c'], stockAccount[CASH_KEY])
            stockAccount[CASH_KEY] = costDiff
            stockAccount[NO_OF_SHARES_KEY] = nShares
    return stockAccount

def Exec_Sell(index, dataSet, stockAccount):
    if  dataSet.loc[index, ind_columns[2]] <= 0:
        if dataSet.loc[index -1, ind_columns[2]] > 0:
            amount = CalcSellReturn(stockAccount[NO_OF_SHARES_KEY], dataSet.loc[index, 'c'])
            stockAccount[CASH_KEY] = stockAccount[CASH_KEY] + amount
            stockAccount[NO_OF_SHARES_KEY] = 0

        if  dataSet.loc[index -1, ind_columns[2]] < 0 and dataSet.loc[index -1, ind_columns[2]] > dataSet.loc[index, ind_columns[2]]:
            amount = CalcSellReturn(stockAccount[NO_OF_SHARES_KEY], dataSet.loc[index, 'c'])
            stockAccount[CASH_KEY] = stockAccount[CASH_KEY] + amount
            stockAccount[NO_OF_SHARES_KEY] = 0
    return stockAccount

def readStockList(path):
    dataFile = open(path, "r")
    dataLines = dataFile.readlines()
    dataLines = [line.rstrip() for line in dataLines]
    dataFile.close()
    return dataLines

def RunForStock(stockName):
    dataFileName = dataFolderPath + '\\'+ stockName + '.txt'
    df = pd.read_json(dataFileName)
    df['t'] = pd.to_datetime(df['t'],unit='s')
    df['t'] = df['t'].dt.date

    if len(df.index) == 0:
        return "{},{},{},{},{}".format(stockName, "Cash", 0,0,0)

    dataSet = CalculateIndices(df)

    stockAccount = {NO_OF_SHARES_KEY:0, CASH_KEY: CASH_VALUE}
    dataSet.head()
    portfolioWorth = 0
    
    for index, row in dataSet.iterrows():
        if index > 0 and row['t'] >= START_TRADING_ON:
            if stockAccount[NO_OF_SHARES_KEY] == 0 :
                stockAccount = Exec_Buy(index, dataSet, stockAccount)
                portfolioWorth = stockAccount[NO_OF_SHARES_KEY] * row['c']
            else:
                stockAccount = Exec_Sell(index, dataSet, stockAccount)
    
    if stockAccount[NO_OF_SHARES_KEY] > 0:        
        return "{},{},{},{},{}".format(stockName, "Hold", portfolioWorth, stockAccount[CASH_KEY], stockAccount[NO_OF_SHARES_KEY] )
    else:
        return "{},{},{},{},{}".format(stockName, "Cash", portfolioWorth, stockAccount[CASH_KEY], stockAccount[NO_OF_SHARES_KEY] )

stockList = readStockList(stockListPath)
results = ''
for stockName in stockList:
    results = results + '\n' + RunForStock(stockName)

file = open(resultFolderPath + '\\BT_MACD_only.csv', 'w')
file.write(results)
file.close()

print('Done')