import pandas as pd
import math
from talib import RSI, BBANDS, MACD
import datetime
from Templating.StocksCommon import CalcBuyCost, CalcSellReturn, BuyQtyForPrice


def BackTestFor(stockName):
    print(stockName)
    fileName = 'D:\\Personal\\share\\data\\'+ stockName +'.txt'
    MACD_FAST = 10
    MACD_SLOW = 21
    MACD_SIGNAL = 5

    INIT_CAPITAL = 100000
    NO_OF_SHARES = 1000
    START_TRADING_ON = datetime.date(2020, 1, 1)

    df = pd.read_json(fileName)
    df['t'] = pd.to_datetime(df['t'],unit='s')
    df['t'] = df['t'].dt.date
    
    print(stockName)

    if len(df.index) == 0:
        return "{},{},{}".format(stockName, "Cash", 0)

    macd, macdSignal, macdHist = MACD(df['c'], MACD_FAST, MACD_SLOW, MACD_SIGNAL)

    #print("Start for stock "+stockName)
    hold = False
    accountHold = INIT_CAPITAL
    portfolioWorth = 0
    nShares = 0
    for index, row in df.iterrows():
        if index > 0 and not math.isnan(macdHist[index-1]) and row['t'] >= START_TRADING_ON:
            if macdHist[index] <= 0:
                if macdHist[index] > macdHist[index-1] and not hold:
                    nShares, costDiff  = BuyQtyForPrice(row['c'], accountHold)
                    accountHold = costDiff
                    #print('Buy -> {} at {} on {}'.format(nShares, str(row['c']), str(row['t'])))
                    hold = True

        if hold and macdHist[index] <= 0:
            if macdHist[index-1] > 0:
                amount = CalcSellReturn(nShares, row['c'])
                accountHold = accountHold + amount
                #print('Sell -> {} at {} on {} : Cash in account {}'.format(nShares, str(row['c']), str(row['t']), accountHold))
                nShares = 0
                hold = False

            if  macdHist[index-1] < 0 and macdHist[index-1] > macdHist[index]:
                amount = CalcSellReturn(nShares, row['c'])
                accountHold = accountHold + amount
                #print('Sell -> {} at {} on {} : Cash in account {}'.format(nShares, str(row['c']), str(row['t']), accountHold))
                nShares = 0
                hold = False
        
        portfolioWorth = nShares * row['c']

    if hold:
        #print("{} Portfolio worth {} remaining cash {}".format(stockName, portfolioWorth, accountHold))
        print("{},{},{}".format(stockName, "Hold", portfolioWorth))
        return "{},{},{}".format(stockName, "Hold", portfolioWorth)
    else:
        #print("{} Cash in hand {}".format(stockName, accountHold))
        print("{},{},{}".format(stockName, "Cash", accountHold))
        return "{},{},{}".format(stockName, "Cash", accountHold)

nfilepath = "D:\\Personal\\share\\BasicData\\ALL-stocks.txt"
nFile = open(nfilepath, "r")
Nlines = nFile.readlines()
Nlines = [line.rstrip() for line in Nlines]
nFile.close()

results = ''
for sLine in Nlines:
    results = results + '\n' + BackTestFor(sLine)

resultFolderPath = 'D:\\Personal\\share\\results'
file = open(resultFolderPath + '\\BackTestResults.csv', 'w')
file.write(results)
file.close()

print('Done')
