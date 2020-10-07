import pandas as pd
import math
from talib import RSI, BBANDS, MACD
import datetime
from Templating.StocksCommon import CalcBuyCost, CalcSellReturn, BuyQtyForPrice
from Templating.BackTestCommon import Exec_Buy, Exec_Sell, NO_OF_SHARES_KEY, CASH_KEY,CASH_VALUE 

START_TRADING_ON = datetime.date(2020, 5, 14)

stockListPath = 'D:\\Personal\\share\\BasicData\\ALL-stocks.txt'
dataFolderPath = 'D:\\Personal\\share\\data'
resultFolderPath = 'D:\\Personal\\share\\results'

ind_columns = ['MACD', 'MACDS', 'MACDH', 'RSI']

def CalculateIndices(df):
    MACD_FAST = 10
    MACD_SLOW = 21
    MACD_SIGNAL = 5
    RSI_T = 10

    macd, macdSignal, macdHist = MACD(df['c'], MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    rsi = RSI(df['c'], RSI_T)

    df[ind_columns[0]] = macd
    df[ind_columns[1]] = macdSignal
    df[ind_columns[2]] = macdHist
    df[ind_columns[3]] = rsi
    return df

def TestForBuySignal(index, dataSet):
    if math.isnan(dataSet.at[index, ind_columns[2]]):
        return False

    if dataSet.at[index, ind_columns[2]] <= 0 and dataSet.at[index, ind_columns[3]] <= 50:
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

    stockAccount = {NO_OF_SHARES_KEY:0, CASH_KEY: CASH_VALUE}

    if len(df.index) == 0:
        return [[stockName, '1900-01-01', 0, 'Idle', stockAccount[CASH_KEY], stockAccount[NO_OF_SHARES_KEY],0]]

    dataSet = CalculateIndices(df)

    detail = []
    for index, row in dataSet.iterrows():
        if index > 0 and row['t'] > START_TRADING_ON:
            #print ("{} {}".format(row['t'], START_TRADING_ON))
            if stockAccount[NO_OF_SHARES_KEY] == 0 :
                if TestForBuySignal(index, dataSet):
                    stockAccount = Exec_Buy(row['c'], stockAccount)                    
                    detail.append([stockName, row['t'], row['c'], 'Buy', stockAccount[CASH_KEY], stockAccount[NO_OF_SHARES_KEY], stockAccount[NO_OF_SHARES_KEY]*row['c']])
            elif TestForSellSignal(index, dataSet):
                stockAccount = Exec_Sell(row['c'], stockAccount)
                detail.append([stockName, row['t'], row['c'], 'Sell', stockAccount[CASH_KEY], stockAccount[NO_OF_SHARES_KEY], stockAccount[NO_OF_SHARES_KEY]*row['c']])
    return detail

# Run Main
stockList = readStockList(stockListPath)
signalData = []
signalSummery = []

for stockName in stockList:
    tmpSignalData =  RunForStock(stockName)
    signalData = signalData + tmpSignalData
    if len(tmpSignalData) > 0 :
        signalSummery.append(tmpSignalData[-1])

columns = ['Stock','Date','Price','Action','Cash','Shares','Value']
data = pd.DataFrame(signalData, columns=columns)
summary = pd.DataFrame(signalSummery, columns=columns)

data.to_csv(resultFolderPath + '\\BT_MACD_RSI.csv')
summary.to_csv(resultFolderPath + '\\BT_MACD_RSI_Sum.csv')

print('Done ')