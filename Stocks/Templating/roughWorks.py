import pandas as pd
import numpy as np
import math
from talib import RSI, BBANDS, MACD
import datetime
import matplotlib.pyplot as plt

plt.style.use('ggplot')

stockCode = 'TILE.N0000'
dataFolderPath = 'D:\\Personal\\share\\data'

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

print(df.tail(20))

series = df[['date']].dropna()
series2 = df['date'].dropna().tolist()
locations = [0, 25, 50 , 100, 125, 150]
data = df.iloc[locations]['date'].tolist()
print(data)