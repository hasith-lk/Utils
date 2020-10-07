import numpy as np
import json
from talib import EMA
import datetime

nfilepath = "D:\\Personal\\share\\BasicData\\All-stocks.txt"
savefolderPath = "D:\\Personal\\share\\data"
resultFolderPath = 'D:\\Personal\\share\\results'
resultsFile = "resultsEMA.txt"

# index settings
EMA_1 = 45
EMA_2 = 15
EMA_3 = 5

def calculateIndex(filePath):
    File = open(filePath, "r")
    text = File.read()
    text = text.rstrip()
    File.close()

    OHLCV = json.loads(text)

    timePeriod = OHLCV['t']
    if timePeriod:
        nClose = np.array(OHLCV['c'])
        ema1 = EMA(nClose, EMA_1)
        ema2 = EMA(nClose, EMA_2)
        ema3 = EMA(nClose, EMA_3)
        return ema1, ema2, ema3
    else:
        return 0, 0, 0

nFile = open(nfilepath, "r")
Nlines = nFile.readlines()
Nlines = [line.rstrip() for line in Nlines]
nFile.close()

values='SHARE,EMA1,EMA2,EMA3\n'
for sLine in Nlines:
    sFileName = savefolderPath + '\\'+ sLine + '.txt'
    try:
        ema1,ema2,ema3 = calculateIndex(sFileName)
        values = values + sLine + ',' + str(ema1[-1]) + ',' + str(ema2[-1]) + ',' + str(ema3[-1]) +'\n'
        
        print(sLine + ' : Success')    
    except:
        print(sLine + ' : Failed') 


file = open(resultFolderPath + '\\' + resultsFile, 'w')
file.write(values)
file.close()

print('done')