import pandas as pd
import numpy as np
import math
from talib import RSI, BBANDS, MACD
import datetime
import matplotlib.pyplot as plt

plt.style.use('ggplot')

stockCode = 'TILE.N0000'
dataFolderPath = 'D:\\Personal\\share\\data'

def CheckMaTrend(shareCode):
    dataFileName = dataFolderPath + '\\'+ stockCode + '.txt'
    df = pd.read_json(dataFileName)
    df['t'] = pd.to_datetime(df['t'], unit='s')
    df['date'] = df['t'].dt.date
    df.set_index(['date'], inplace = True, drop=False)
    df.index = pd.to_datetime(df.index)

    df['10'] = round(df['c'].rolling(window=10).mean(), 4)
    df['21'] = round(df['c'].rolling(window=21).mean(), 4)
    slope = pd.Series(np.gradient(df['21']), df.index, name='slop')
    df['slop'] = slope

    dateRangeIndex = pd.date_range(start=min(df.index), end=max(df.index), freq='D')
    #print(dateRangeIndex)
    df = df.reindex(dateRangeIndex)

    print(df.head(20))
    slope = df[['slop']].dropna()
    #slop_val = df.loc[(max(df.index)-5):max(df.index), list('slop')].sum()
    print(slope[-5:].sum())

    plt.figure(figsize=(16,8))
    plt.title('Moving Average of {}'.format(shareCode))
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Close Price MA', fontsize=12)
    
    #plt.plot(df['c'])
    plt.plot( range(df[['10']].dropna().size), df[['10']].dropna())
    plt.plot( range(df[['21']].dropna().size), df[['21']].dropna())
    #plt.plot( range(df[['slop']].dropna().size), df[['slop']].dropna())
    plt.legend(['10', '21', 'Slop'], loc='lower right')
    plt.xlim(0)
    locs, labels = plt.xticks()
    locs = np.arange(0, 225, step=25)
    labs = df.iloc[locs]['date'].dropna().tolist()
    print(labs)
    plt.xticks(locs, labels, rotation=90)
    
    plt.show()

CheckMaTrend(stockCode)



""" import pandas as pd
import matplotlib.pyplot as plt

pd.np.random.seed(1234)
idx = pd.date_range(end='31/12/2020', periods=10, freq='D')
vals = pd.Series(pd.np.random.randint(1, 10, size=idx.size), index=idx)
vals.iloc[4:8] = pd.np.nan
print (vals)

plt.figure(figsize=(16,8))
plt.plot(range(vals.dropna().size), vals.dropna())
locs, labels = plt.xticks()
plt.xticks(locs, vals.dropna().index.date.tolist())
#plt.plot(df[['10', '21']])
plt.legend(['10', '21'], loc='lower right')
plt.show()

fig, ax = plt.subplots()
ax.plot(range(vals.dropna().size), vals.dropna())
ax.set_xticklabels(vals.dropna().index.date.tolist())
fig.autofmt_xdate()
fig.show() """