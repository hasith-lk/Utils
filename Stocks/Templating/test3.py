from talib import RSI, BBANDS, MACD

import pandas as pd
df = pd.read_json('D:\\Personal\\share\\data\\JKH.N0000.txt')
df['t'] = pd.to_datetime(df['t'], unit='s')
df['t'] = df['t'].dt.date
df.set_index('t', inplace = True)

df.drop('s', axis=1, inplace=True)

resultFolderPath = 'D:\\Personal\\share\\Annotate\\JKH.N0000.csv'
df.to_csv(resultFolderPath)
df.head()

import pandas as pd
data1 = ['JKH', '2010-10-21', 'Buy']
data2 = ['MAL', '2015-10-21', 'Sell']
data3 = ['AEL', '2017-10-21', 'Hold']

main = []
main.append(data1)
main.append(data2)
main.append(data3)

print(main)

columns = ['Stock','Date', 'Signal']
data = pd.DataFrame(main, columns=columns)
data.head()

import os
from datetime import datetime

dataFolder = 'D:\\Personal\\share\\data'
if  os.listdir(dataFolder):
    dateTimeObj = datetime.now()
    backupFolderName=dataFolder + "_{}_{}_{}_{}_{}_{}".format(dateTimeObj.year,dateTimeObj.month,dateTimeObj.day,
    dateTimeObj.hour,dateTimeObj.minute,dateTimeObj.second);
    os.rename(dataFolder,backupFolderName)
    os.makedirs(dataFolder)
print('done')



import plotly.graph_objects as go
fig = go.Figure(go.Scatter(x=[1, 2], y=[1, 0]))

def zoom(layout, xrange):
    print('XXXXX')

fig.layout.on_click(zoom, 'xaxis.range')

fig.show()    
print('done')



