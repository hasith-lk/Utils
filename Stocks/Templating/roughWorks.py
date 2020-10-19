import pandas as pd
import numpy as np
import math
from talib import RSI, BBANDS, MACD
import datetime
import matplotlib.pyplot as plt

plt.style.use('ggplot')

stockCode = 'COMB.N0000'
dataFolderPath = 'D:\\Personal\\share\\data'

dataFileName = dataFolderPath + '\\'+ stockCode + '.txt'
df = pd.read_json(dataFileName)
df['t'] = pd.to_datetime(df['t'], unit='s')
df['date'] = df['t'].dt.date
df.set_index(['date'], inplace = True, drop=False)
df.index = pd.to_datetime(df.index)

plt.figure(figsize = (10,8))
plt.plot(range(df.shape[0]),(df['l']+df['h'])/2.0)
plt.xticks(range(0,df.shape[0],500),df['date'].loc[::500],rotation=45)
#plt.xlabel('Date',fontsize=18)
plt.ylabel('Mid Price',fontsize=18)
plt.show()

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


import numpy as np
N = 10
result = np.array(N).reshape(1,-1)
print(result)
i= 100
X_train = np.array(range(len(df['c'][i-N:i])))
print((df['c'][i-N:i]))


df.tail(10)
#shifting n rows up of a given variable
df['c_shift']=df['c'].shift(periods=-3)
df.tail(10)