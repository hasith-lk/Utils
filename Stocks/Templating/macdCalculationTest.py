from talib import RSI, BBANDS, MACD
import pandas as pd
import matplotlib.pyplot as plt

stockCode = 'JKH.N0000'
dataFolderPath = 'D:\\Personal\\share\\data'

MACD_FAST = 10
MACD_SLOW = 21
MACD_SIGNAL = 8


dataFileName = dataFolderPath + '\\'+ stockCode + '.txt'
df = pd.read_json(dataFileName)
df['t'] = pd.to_datetime(df['t'], unit='s')
df['date'] = df['t'].dt.date
df.set_index(['date'], inplace = True, drop=False)


macd, macdSignal, macdHist = MACD(df['c'], MACD_FAST, MACD_SLOW, MACD_SIGNAL - 1)

#df = df.sort_index(ascending=0)
#df['backward_ewm'] = df['c'].ewm(span=20,min_periods=0,adjust=False).mean()
df = df.sort_index()

df['ewm10'] = round( df['c'].ewm(span=MACD_FAST, adjust=False).mean(), 4)
df['ewm21'] = round(df['c'].ewm(span=MACD_SLOW, adjust=False).mean(), 4)
df['u_macd'] = round(df['ewm10'] - df['ewm21'] , 4) # 10 - 21 diff
df['u_macds'] = round(df['u_macd'].ewm(span=MACD_SIGNAL, adjust=False).mean(),4)
df['u_macds2'] = round(df['u_macd'].rolling(window=MACD_SIGNAL).mean(),4)
df['u_macdh'] = round(df['u_macds2'] - df['u_macd'], 4)


df['macdh'] = macdHist
df['macd'] = macd
df['macds'] = macdSignal

print(df.tail(10))

resultFolderPath = 'D:\\Personal\\share\\results'
df.to_csv(resultFolderPath + '\\Calculation.csv')


'''
fig, ax = plt.subplot()
plt.plot(df['date'], df['u_macdh'], label='U MACD H')
plt.plot(df['date'], df['u_macd'], label='U MACD')

plt.plot(df['date'], df['macd'], label='MACD')
plt.plot(df['date'], df['macdh'], label='MACDH')
plt.legend(loc="lower left")
plt.show()
'''