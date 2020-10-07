import numpy as np
import json
from talib import RSI, BBANDS, MACD, EMA
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime


# index settings
RSI_T = 10
MACD_FAST = 10
MACD_SLOW = 20
MACD_SIGNAL = 5
EMA_SLOW = 45
EMA_MEDIUM = 15
EMA_FAST = 5

def calculateIndex(filePath):
    File = open(filePath, "r")
    text = File.read()
    text = text.rstrip()
    File.close()

    OHLCV = json.loads(text)
    timePeriod =np.array(OHLCV['t'])
    nClose = np.array(OHLCV['c'])

    rsi = RSI(nClose, timeperiod=RSI_T)
    macd, macdSignal, macdHist = MACD(nClose, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    emaSlow = EMA(nClose, EMA_SLOW)
    emaMedium = EMA(nClose, EMA_MEDIUM)
    emaFast = EMA(nClose, EMA_FAST)

    #print("RSI (first 10 elements)\n", rsi[10:20])
    return timePeriod, rsi, macd, macdSignal, macdHist, emaSlow, emaMedium, emaFast

def DrawChart(stock, timePeriod, imageSavePath, rsi, macd, macdSignal, macdHist, emaSlow, emaMedium, emaFast):
    # Prepare plot
    plt.style.use('ggplot')
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    fig.suptitle(stock)

    #size plot
    fig.set_size_inches(15,30)    

    # Chart Commons
    chartDates = mdates.epoch2num(timePeriod)
    years = mdates.YearLocator()   # every year
    months = mdates.MonthLocator()  # every month
    auto = mdates.AutoDateLocator()
    years_fmt = mdates.AutoDateFormatter(auto)
    dayLoc = mdates.DayLocator()
    ax1.xaxis.set_major_locator(auto)
    ax1.xaxis.set_major_formatter(years_fmt)
    ax1.xaxis.set_minor_locator(dayLoc)

    # RSI chart
    rsiSupport = 40
    rsiResistance = 70

    ax1.set_ylabel('RSI ')
    ax1.plot(chartDates, rsi,  color='b', label='RSI')
    ax1.axhline(y=rsiSupport, c='b')
    ax1.axhline(y=50, c='black')
    ax1.axhline(y=rsiResistance, c='b')
    ax1.set_ylim([0,100])
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels)

    # MACD
    ax2.set_ylabel('MACD ')
    ax2.plot(chartDates, macd, color='b', label='Macd')
    ax2.plot(chartDates, macdSignal, color='r', label='Signal')
    ax2.plot(chartDates, macdHist, color='y', label='Hist')
    ax2.axhline(0, lw=2, color='0')
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels)

    # EMA
    ax3.set_ylabel('EMA ')
    ax3.plot(chartDates, emaSlow, color='b', label='Slow')
    ax3.plot(chartDates, emaMedium, color='r', label='Medium')
    ax3.plot(chartDates, emaFast, color='y', label='Fast')
    ax3.axhline(0, lw=2, color='0')
    handles, labels = ax3.get_legend_handles_labels()
    ax3.legend(handles, labels)

    fig.savefig(imageSavePath, bbox_inches='tight', pad_inches=0)
    fig.close()

    #plt.show()

shareType='X'
drawCharts=True
nfilepath = "D:\\personal\\share\\" + shareType + "-stocks.txt"

savefolderPath = "D:\\personal\\share\\data"
nFile = open(nfilepath, "r")
Nlines = nFile.readlines()
Nlines = [line.rstrip() for line in Nlines]
nFile.close()

values = 'STOCK,RSI,MACD,MACDSIGNAL,MACDHISTO,EMASLOW,EMAMEDIUM,EMAFAST\n'
for sLine in Nlines:
    sFileName = savefolderPath + '\\'+ sLine + '.txt'
    print(sFileName)
    try:
        timePeriod, rsi, macd, macdSignal, macdHist, emaSlow, emaMedium, emaFast = calculateIndex(sFileName)
        if len(rsi) > 1 :
            values = values + '{STOCK},{RSI},{MACD},{MACDSIGNAL},{MACDHIST},{EMASLOW},{EMAMEDIUM},{EMAFAST}\n'.format(
                STOCK=sLine, RSI=rsi[-1], MACD=macd[-1], MACDSIGNAL=macdSignal[-1], MACDHIST=macdHist[-1], EMASLOW=emaSlow[-1], EMAMEDIUM=emaMedium[-1], EMAFAST=emaFast[-1])
            if drawCharts:
                imageSavePath = savefolderPath + '\\' +sLine+'.svg'
                DrawChart(sLine, timePeriod, imageSavePath, rsi, macd, macdSignal, macdHist, emaSlow, emaMedium, emaFast)
            print(sLine + ' : Success')
        else:
            print(sLine + ' : Failed')
    except:
        print(sLine + ' : Failed')

resultFolderPath = 'D:\\Personal\\share\\results'
file = open(resultFolderPath + '\\IndicatorResults' + shareType + '.csv', 'w')
file.write(values)
file.close()

print('Done')
