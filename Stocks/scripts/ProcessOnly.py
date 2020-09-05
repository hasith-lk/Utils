import numpy as np
import json
from talib import RSI, BBANDS, MACD
import datetime

nfilepath = "D:\\Personal\\share\\BasicData\\ALL-stocks.txt"
savefolderPath = "D:\\Personal\\share\\results"

# index settings
RSI_T = 10
MACD_FAST = 10
MACD_SLOW = 21
MACD_SIGNAL = 5

def calculateIndex(filePath):
    File = open(filePath, "r")
    text = File.read()
    text = text.rstrip()
    File.close()

    OHLCV = json.loads(text)

    timePeriod = OHLCV['t']

    if timePeriod:
        nClose = np.array(OHLCV['c'])
        rsi = RSI(nClose, timeperiod=RSI_T)
        macd, macdSignal, macdHist = MACD(nClose, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
        #print("RSI (first 10 elements)\n", rsi[-1])
        return rsi, macd, macdSignal, macdHist
    else:
        return 100, 0, 0, 0

nFile = open(nfilepath, "r")
Nlines = nFile.readlines()
Nlines = [line.rstrip() for line in Nlines]
nFile.close()

values='SHARE,RSI,MACD_DIV,MACD,MACD_SIG\n'
for sLine in Nlines:
    sFileName = savefolderPath + '\\'+ sLine + '.txt'
    try:
        rsi, macd, macdSignal, macdHisto = calculateIndex(sFileName)
        if len(rsi) > 1 :
            values = values +sLine + ',' + str(rsi[-1])+',' + str(macdHisto[-1]) +','+ str(macd[-1]) +','+ str(macdSignal[-1]) +'\n'
        else:
            values = values +sLine + ' RSI 0 \n'
        
        print(sLine + ' : Success')    
    except:
        print(sLine + ' : Failed') 


resultFolderPath = 'D:\\Personal\\share\\results'
file = open(resultFolderPath + '\\IndicatorResults.csv', 'w')
file.write(values)
file.close()

print('done')