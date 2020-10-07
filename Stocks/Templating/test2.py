import pandas as pd
import numpy as np
from talib import RSI, BBANDS
import matplotlib.pyplot as plt


price = pd.read_json('D:\\Personal\\share\\data\\JKH.N0000.txt')

price = price.dropna()
price['t'] = pd.to_datetime(price['t'],unit='s')
price.set_index('t', inplace = True)

close = price['c'].values
up, mid, low = BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
rsi = RSI(close, timeperiod=14)
print("RSI (first 10 elements)\n", rsi[0:24])
price['RSI'] = rsi
price.head(20)


def bbp(price):
    up, mid, low = BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    bbp = (price['c'] - low) / (up - low)
    return bbp

index = price.index
max_holding = 100


holdings = pd.DataFrame(index=price.index, data={'Holdings': np.array([np.nan] * index.shape[0])})
holdings.loc[((price['RSI'] < 30) & (price['BBP'] < 0)), 'Holdings'] = max_holding
holdings.loc[((price['RSI'] > 70) & (price['BBP'] > 1)), 'Holdings'] = 0
holdings.ffill(inplace=True)
holdings.fillna(0, inplace=True)
holdings['Order'] = holdings.diff()
holdings.dropna(inplace=True)

fig, (ax0, ax1, ax2) = plt.subplots(3, 1, sharex=True, figsize=(12, 8))
ax0.plot(index, price['c'], label='Close')
ax0.set_xlabel('Date')
ax0.set_ylabel('c')
ax0.grid()
for day, holding in holdings.iterrows():
    order = holding['Order']
    if order > 0:
        ax0.scatter(x=day, y=price.loc[day, 'c'], color='green')
    elif order < 0:
        ax0.scatter(x=day, y=price.loc[day, 'c'], color='red')

ax1.plot(index, price['RSI'], label='RSI')
ax1.fill_between(index, y1=30, y2=70, color='#adccff', alpha='0.3')
ax1.set_xlabel('Date')
ax1.set_ylabel('RSI')
ax1.grid()

ax2.plot(index, price['BB_up'], label='BB_up')
ax2.plot(index, price['c'], label='Close')
ax2.plot(index, price['BB_low'], label='BB_low')
ax2.fill_between(index, y1=price['BB_low'], y2=price['BB_up'], color='#adccff', alpha='0.3')
ax2.set_xlabel('Date')
ax2.set_ylabel('Bollinger Bands')
ax2.grid()

fig.tight_layout()
plt.show()